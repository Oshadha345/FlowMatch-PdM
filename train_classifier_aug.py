import sys

from train_classifier import main


LEGACY_ARG_ALIASES = {
    "--run_id": "--source_run_id",
    "--output_run_id": "--run_id",
}


if __name__ == "__main__":
    argv = sys.argv[1:]
    raw_mode = False
    for index, token in enumerate(argv):
        if token == "--model" and index + 1 < len(argv) and argv[index + 1].strip().lower() == "raw":
            raw_mode = True
        if token.startswith("--model=") and token.split("=", 1)[1].strip().lower() == "raw":
            raw_mode = True

    remapped = []
    skip_next = False

    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if token == "--model" and index + 1 < len(argv) and argv[index + 1].strip().lower() == "raw":
            skip_next = True
            continue
        if token.startswith("--model=") and token.split("=", 1)[1].strip().lower() == "raw":
            continue

        if token in LEGACY_ARG_ALIASES:
            if raw_mode and token == "--run_id":
                remapped.append(token)
                if index + 1 < len(argv):
                    remapped.append(argv[index + 1])
                    skip_next = True
                continue
            remapped.append(LEGACY_ARG_ALIASES[token])
            if index + 1 < len(argv):
                remapped.append(argv[index + 1])
                skip_next = True
            continue

        remapped.append(token)

    sys.argv = [sys.argv[0], *remapped]
    main()
