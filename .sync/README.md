# opt deploy

Скрипты в этой папке поднимают `LoraBackdoorDetection` на SSH-хосте `opt`.

## Что делает `deploy.sh`

1. Синхронизирует локальный репозиторий на `opt` через `rsync`
2. Создаёт `venv` на удалённой машине
3. Устанавливает зависимости из `requirements.txt`
4. Записывает `HF_TOKEN` в удалённый `.env`

## Быстрый запуск

```bash
cd /home/weshi/LoraBackdoorDetection
HF_TOKEN=hf_xxx ./.sync/deploy.sh
```

## Отдельные шаги

```bash
./.sync/sync_to_opt.sh
HF_TOKEN=hf_xxx ./.sync/bootstrap_opt.sh
```

## Переменные

```bash
REMOTE_HOST=opt
REMOTE_REPO_DIR=/home/vskate/LoraBackdoorDetection
SYNC_RESULTS=0
SYNC_ANALYSIS=0
SYNC_OUTPUTS=0
```

По умолчанию копируются только код и конфиги.
Тяжёлые каталоги `results/`, `resultsFinal/`, `projection_analysis/` и `output_*` не копируются.

Если нужно перетащить адаптеры для полного воспроизведения evaluation, включи:

```bash
SYNC_OUTPUTS=1 ./.sync/sync_to_opt.sh
```
