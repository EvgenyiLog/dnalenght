# Прочитать все строки из flake8-отчета
Get-Content .\errorsflake8.txt | ForEach-Object {
    if ($_ -match "^(.*?):(\d+):\d+:\s+([A-Z]\d+)\s") {
        $filename = $matches[1]
        $line = [int]$matches[2]
        $code = $matches[3]

        Write-Host "Fixing $filename line $line [$code]..."

        # ВАЖНО: --line-range два отдельных аргумента
        autopep8 --in-place --line-range $line $line --select $code $filename
    }
}