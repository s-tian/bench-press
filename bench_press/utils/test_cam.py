import argparse

import cv2


def main(cams):
    # Capture indices go up by 2 with the OmniTact setup
    caps = [cv2.VideoCapture(i * 2) for i in cams]

    while True:
        frames = [cap.read()[1] for cap in caps]
        ims = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]

        # Display the resulting frame
        for i, frame in enumerate(ims):
            cv2.imshow(f'frame{i}', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test cameras')
    parser.add_argument('-c', '--cams', help='List of cam indices to open', nargs='+', action='store', default='0 2 4')
    args = parser.parse_args()
    main(args.cams)
