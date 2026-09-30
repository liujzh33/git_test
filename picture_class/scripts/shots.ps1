param([string[]]$Ids, [int]$W = 1600, [int]$H = 900)
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$out = Join-Path $PSScriptRoot "..\shots"
New-Item -ItemType Directory -Force $out | Out-Null
foreach ($id in $Ids) {
  $args = @("--headless=new", "--disable-gpu", "--hide-scrollbars", "--user-data-dir=$env:TEMP\edgeshot_$(Get-Random)",
            "--window-size=$W,$H", "--virtual-time-budget=6000", "--screenshot=$out\$id.png",
            "http://localhost:8765/index.html?only=$id")
  Start-Process -FilePath $edge -ArgumentList $args -Wait -NoNewWindow -RedirectStandardError "$env:TEMP\edge_err.txt" | Out-Null
  "{0}: {1} KB" -f $id, [math]::Round((Get-Item "$out\$id.png").Length / 1KB)
}
