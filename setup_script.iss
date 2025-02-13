#define MyAppName "Transformer Prediction System"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "DATSAN"
#define MyAppURL "https://www.datsan.com.tr"
#define MyAppExeName "Transformer Prediction System.exe"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
AppId={{A45B7D1A-4B3C-4D8F-9A1E-3A8C8F2B5E9D}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
OutputDir=Output
OutputBaseFilename=TransformerPredictionSystem_Setup
SetupIconFile=datsanstlogo.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "turkish"; MessagesFile: "compiler:Languages\Turkish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\Transformer Prediction System\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "ag_yassi_tel.joblib"; DestDir: "{app}"; Flags: ignoreversion
Source: "YG_Emaye.joblib"; DestDir: "{app}"; Flags: ignoreversion
Source: "folyo.joblib"; DestDir: "{app}"; Flags: ignoreversion
Source: "logo.jpeg"; DestDir: "{app}"; Flags: ignoreversion
Source: "datsanstlogo.ico"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\datsanstlogo.ico"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\datsanstlogo.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;

[UninstallDelete]
Type: filesandordirs; Name: "{app}" 