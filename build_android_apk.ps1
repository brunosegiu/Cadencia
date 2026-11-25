$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$sdk = "$env:LOCALAPPDATA\Android\Sdk"
$buildTools = "$sdk\build-tools\34.0.0"
$platformJar = "$sdk\platforms\android-34\android.jar"
$ndkBin = "$sdk\ndk\26.1.10909125\toolchains\llvm\prebuilt\windows-x86_64\bin"
$jdkHome = "$sdk\jdk-17\jdk-17.0.20.1+1"

$env:JAVA_HOME = $jdkHome
$env:PATH = "$jdkHome\bin;$ndkBin;$buildTools;$env:PATH"
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER = "$ndkBin\aarch64-linux-android26-clang.cmd"
$env:CC_aarch64_linux_android = "$ndkBin\aarch64-linux-android26-clang.cmd"
$env:CXX_aarch64_linux_android = "$ndkBin\aarch64-linux-android26-clang++.cmd"
$env:AR_aarch64_linux_android = "$ndkBin\llvm-ar.exe"
$env:CARGO_TARGET_AARCH64_LINUX_ANDROID_RUSTFLAGS = "-C link-arg=-Wl,-z,max-page-size=16384"

Write-Host "==> 1. Compiling Rust aarch64-linux-android shared library (libcadencia_lib.so)..."
cargo build --offline --release --lib --target aarch64-linux-android --no-default-features

$soPath = "target\aarch64-linux-android\release\libcadencia_lib.so"
if (-not (Test-Path $soPath)) {
    throw "Failed to locate compiled $soPath"
}

$apkBuildDir = "target\android_apk"
if (Test-Path $apkBuildDir) {
    Remove-Item -Recurse -Force $apkBuildDir
}
New-Item -ItemType Directory -Path "$apkBuildDir\staging\lib\arm64-v8a" -Force | Out-Null

Write-Host "==> 2. Linking AndroidManifest.xml with aapt2..."
& "$buildTools\aapt2.exe" link `
    -o "$apkBuildDir\unaligned.apk" `
    --manifest "android\AndroidManifest.xml" `
    -I "$platformJar"

Write-Host "==> 3. Stripping debug symbols & packaging lib/arm64-v8a/libcadencia_lib.so..."
& "$ndkBin\llvm-strip.exe" --strip-unneeded -o "$apkBuildDir\staging\lib\arm64-v8a\libcadencia_lib.so" "$soPath"

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::Open("$PWD\$apkBuildDir\unaligned.apk", [System.IO.Compression.ZipArchiveMode]::Update)
try {
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
        $zip,
        "$PWD\$apkBuildDir\staging\lib\arm64-v8a\libcadencia_lib.so",
        "lib/arm64-v8a/libcadencia_lib.so",
        [System.IO.Compression.CompressionLevel]::NoCompression
    ) | Out-Null
} finally {
    $zip.Dispose()
}

Write-Host "==> 4. Aligning APK with zipalign..."
& "$buildTools\zipalign.exe" -f -p 4 "$apkBuildDir\unaligned.apk" "$apkBuildDir\aligned.apk"

$keystore = "target\debug.keystore"
if (-not (Test-Path $keystore)) {
    Write-Host "==> 5. Generating debug keystore..."
    & "$jdkHome\bin\keytool.exe" -genkeypair `
        -keystore "$keystore" `
        -storepass android `
        -keypass android `
        -alias androiddebugkey `
        -keyalg RSA `
        -keysize 2048 `
        -validity 10000 `
        -dname "CN=Android Debug,O=Android,C=US"
}

$finalApk = "cadencia.apk"
Write-Host "==> 6. Signing APK with apksigner -> $finalApk..."
& "$buildTools\apksigner.bat" sign `
    --ks "$keystore" `
    --ks-pass pass:android `
    --key-pass pass:android `
    --ks-key-alias androiddebugkey `
    --out "$finalApk" `
    "$apkBuildDir\aligned.apk"

& "$buildTools\apksigner.bat" verify --verbose "$finalApk"
Write-Host "==> APK successfully built and signed at: $PWD\$finalApk"
