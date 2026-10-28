# Delt jobbkø – kort bruksark (16. oktober 2026)

**Status:** J1–J3 er bestått; Main J4 er **exact-39 PASS** på `24fdaace` (CPU float32, AC52, 20:54 UTC): original pause/600 s gjenopptakelse og avbrudd/600 s gjenvinning, publisering, collect og alle 24 observerte prosessinkarnasjoner avsluttet. **J4 på den andre verten er fortsatt PENDING før den kan delta:** N9/N10 er fulgt; den eksisterende «DarSIA Git worker console» leste seriens UNC-manifest med samme SHA og størrelse som referanseverten 21:55 UTC. Oppgaven og begge observerte prosessinkarnasjoner er avsluttet. Ingen beregning eller claim er forsøkt på den andre verten. Målt ledig plass er 49,09 GiB; hele AC52 med midlertidige filer, kode og uendret 40 GiB reserve krever 73,20 GiB før kloning (24,11 GiB mangler). Et tilstrekkelig eksisterende lokalt mål eller frigjort plass må avklares før den andre verten deltar; ren utsjekking, eget kart, miljø-/inndataadmisjon og exact-39 gjenstår. Kommandoene her gir ingen ny oppstartstillatelse.

## 1. Samme jobb, hver maskins kart

Bare jobbmappen deles, på ekte SMB/NFS. Synkmapper som OneDrive/Dropbox og ukjent transport avvises. Begge maskiner må ha rent Git-arbeidstre på jobbens eksakte commit og miljølås.

`series.toml` deklarerer `[roots]`: `series = "."`, `raw` og `legacy_results`. De siste to peker i FF_AC på henholdsvis `//<nas>/<share>/Albus/Raw data` og `//<nas>/<share>/Albus/Results`. Nye jobber har `format = "darsia-alias-job-v1"`, `format_version = 1`, låst commit/miljø/SHA, kalibrerings-SHA og én presisjon. Hver inndatareferanse har nøyaktig `root_alias`, `relative_path`, `role`, `sha256`, `bytes`. Ukjent rot stopper opprettelsen før publisering; ingen maskinsti eller kartfilsti legges i jobben.

Hver maskin har **sin egen lokale** `%LOCALAPPDATA%/DarSIA/stores.toml`, utenfor repo og serie. På andre systemer brukes `$XDG_CONFIG_HOME/darsia/stores.toml`, ellers `~/.config/darsia/stores.toml`. Referanseverten sitt installerte kart er:

```toml
[machine]
id = "<referansevert>"
[stores.FF_AC]
series = "//<nas>/<share>/FF_AC/DarSIA"
raw = "//<nas>/<share>/Albus/Raw data"
legacy_results = "//<nas>/<share>/Albus/Results"
results = "C:/FFAC_cache/darsia_results"
```

Den andre vertens **planlagte, ennå ikke installerte/verifiserte** eget kart har `id = "<andre vert>"`, de samme tre UNC-inndatarøttene og egen lokal `results = "C:/FFAC_cache/darsia_results"`. Ingenting forhåndskopieres. Jobs → Machine stores/Settings velger kartet; Check availability leser tilgjengelighet uten å starte en jobb.

## 2. Den faste arbeiderkommandoen

Kjør med riktig låst utsjekkings Python-miljø og sett arbeidsmappen til `<utsjekking>/src`, slik at Python laster denne koden. Konkrete J4-eksempler: første jobb er allerede ferdig; eksemplet for den andre verten er kun planlagt; UNC-tilgangen er bekreftet, mens eget kart, riktig commit og de øvrige J4-portene gjenstår. For senere godkjent arbeid hentes jobbstien fra den faktiske delte jobben.

```text
python -m darsia.fluidflower.worker --job "//<nas>/<share>/FF_AC/DarSIA/jobs/j4-main-ac52-alias-20261016-v3" --backend cpu --workers 1 --machine <referansevert> --stores "C:/Users/<user>/AppData/Local/DarSIA/stores.toml" --watchdog
python -m darsia.fluidflower.worker --job "//<nas>/<share>/FF_AC/DarSIA/jobs/j4-<andre vert>-ac52-alias-20261016-v1" --backend cpu --workers 1 --machine <andre vert> --stores "C:/Users/<user>/AppData/Local/DarSIA/stores.toml" --once
```

Før atomisk claim må alle oppgavens inndata løses og ha riktig størrelse. Etter claim hashes hver unik fysisk fil én gang før første bilde; de opprinnelige vitenskapelige SHA-kontrollene beholdes. Avvist kjøring får grunn og ny attempt-identitet; samme arbeider prøver den ikke blindt igjen. Pause/stopp fra Jobs skjer etter et committed bilde. Leien er 600 sekunder uten faktisk framdrift, målt med serverens fil-mtime; omstart omgår ikke ventingen. Startkvittering og SSH-exit er ikke framdrift. Den andre verten bruker egen innlogget konsoll/RDP eller en uttrykkelig godkjent eksisterende innloggingsregistrering; ingen ny tjeneste eller generert startkode.

## 3. Produkter og collect

Produkter skrives lokalt i `<results>/<jobb>/<kjøring>/.attempts/<token>`. Delte framdriftsfiler er SHA-bundne bytekopier av små commit-filer. Fullføring publiserer uendret Main-format `completed.json`, manifest, skalarer, kvitteringer og `products_index.json` med produkt-ID, maskin, lokal opphavssti, SHA og byteantall. Master-visningen leser publiserte filer. Store kart blir lokale; `publish` kopierer bare eksisterende, merkede `film`/`lossless_export`-produkter til egen delt bundle og lager ingen produkter.

```text
python -m darsia.fluidflower.products collect --index "<faktisk delt products_index.json>" --index-sha256 "<SHA256>" --index-bytes <heltall> --product-id "mass_table.csv" --target "<ny lokal målmappe>" --stores "<eget lokalt stores.toml>"
```

Gjenta `--product-id` for flere produkter. Bruk `--published-bundle "<delt publication.json>"` for allerede publiserte store produkter, ellers eksplisitt `--source "<produkt-ID>=<tilgjengelig fil>"`. Collect kontrollerer SHA/størrelse, søker ikke etter filer og følger aldri automatisk en annen maskins C:-sti. Fullføring og faktisk prosessavslutning er separate bevis. Gamle absolutte runde-4-jobber beholder originalflyten og merkes `legacy_paths=true`.
