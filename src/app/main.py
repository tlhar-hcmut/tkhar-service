import time


def xprint(*values: object):
    print(*values, flush=True)


def main():
    xprint("Start")
    time.sleep(10)
    xprint("Stop")


if __name__ == "__main__":
    main()
