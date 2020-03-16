import time

from gelsight_tb.tb_control.testbench_control import TestBench


def main():
    tb = TestBench('/dev/ttyACM0')
    tb.flip_x_reset()

    while not tb.ready():
        time.sleep(0.1)
        tb.update()

    tb.start()

    while tb.busy():
        tb.update()

    while True:
        data = tb.req_data()
        forces = [data['force_{}'.format(i)] for i in range(1, 5)]
        st = ''
        for i, force in enumerate(forces):
            st += ' load cell {}: {}'.format(i + 1, force)
        print(st)
        time.sleep(0.5)


if __name__ == '__main__':
    main()
