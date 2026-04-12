import sys

from train_classifier import main


LEGACY_ARG_ALIASES = {
    "--run_id": "--source_run_id",
}


if __name__ == "__main__":
    argv = sys.argv[1:]
    remapped = []
    skip_next = False

    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if token in LEGACY_ARG_ALIASES:
            remapped.append(LEGACY_ARG_ALIASES[token])
            if index + 1 < len(argv):
                remapped.append(argv[index + 1])
                skip_next = True
            continue

        remapped.append(token)

    sys.argv = [sys.argv[0], *remapped]
    main()
