Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pyw -3 """ & Replace(WScript.ScriptFullName, "NoAnimals.vbs", "noanimals_tray.pyw") & """", 0, False
