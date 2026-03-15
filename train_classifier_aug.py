import sys

from train_classifier import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    remapped = []
    skip_next = False

    for index, token in enumerate(argv):
        if skip_next:
            skip_next = False
            continue

        if token == "--run_id":
            remapped.append("--source_run_id")
            if index + 1 < len(argv):
                remapped.append(argv[index + 1])
                skip_next = True
            continue

        remapped.append(token)

    sys.argv = [sys.argv[0], *remapped]
    main()
