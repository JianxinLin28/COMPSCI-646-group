# -------------------------------
# Windows setup_java.ps1 under user folder
# -------------------------------

# Get current username
$userName = $env:USERNAME
$userJdkDir = "C:\Users\$userName\jdk-23.0.1"

Write-Host "Setting up Java 23.0.1 in $userJdkDir"

# 1️⃣ Download JDK
$jdkUrl = "https://download.oracle.com/java/23/archive/jdk-23.0.1_windows-x64_bin.zip"
$jdkZip = "$env:TEMP\jdk-23.0.1_windows-x64_bin.zip"
Invoke-WebRequest -Uri $jdkUrl -OutFile $jdkZip

# 2️⃣ Extract JDK to user folder
Expand-Archive -Path $jdkZip -DestinationPath $userJdkDir -Force

# 3️⃣ Remove the zip file
Remove-Item $jdkZip

# 4️⃣ Set environment variables for current session
$env:JAVA_HOME = $userJdkDir
$env:Path = "$env:JAVA_HOME\bin;$env:Path"

# 5️⃣ Optionally, set them permanently for the user
setx JAVA_HOME "$userJdkDir"
# Add JAVA_HOME\bin to user PATH permanently
setx Path "%JAVA_HOME%\bin;%Path%"

# 6️⃣ Verify
Write-Host "Java version:"
java -version
Write-Host "JAVA_HOME:"
echo $env:JAVA_HOME
