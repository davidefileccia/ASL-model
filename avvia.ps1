$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = "C:\asl_env\Scripts\python.exe"
$script = Join-Path $scriptDir "asl_realtime_inference.py"
Set-Location $scriptDir
& $python $script
