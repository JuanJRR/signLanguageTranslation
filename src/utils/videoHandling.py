import math

import cv2

from utils.logger import Logger


class VideoHandling:
    """Functional for video manipulation"""

    def __init__(
        self, pixels: int = 90, aspect_ratio: list = [16, 9], color: str = "GRAY"
    ) -> None:
        """Builder

        :param pixels: Pixel resolution for video rescaling, defaults to 90
        :type pixels: int, optional
        :param aspect_ratio: Video aspect ratio or scaling, defaults to [16, 9]
        :type aspect_ratio: list, optional
        :param color: Color dimension for the video. GRAY for gray scale and RGB for color space (red, green, blue.)
        , defaults to "GRAY"
        :type color: str, optional
        """
        # settings
        logger = Logger()
        self.log = logger.config_logging()

        dim_width = 0
        dim_high = 0
        if aspect_ratio == [16, 9]:
            dim_width = int(pixels * 2)
            dim_high = int(pixels)

        # dim_width = math.floor((pixels / aspect_ratio[1]) * aspect_ratio[0])
        # if dim_width % 2 == 0:
        #     dim_width = dim_width
        # else:
        #     dim_width = dim_width + 1

        # dim_high = math.floor((pixels / aspect_ratio[1]) * aspect_ratio[1])
        # if dim_high % 2 == 0:
        #     dim_high = dim_high
        # else:
        #     dim_high = dim_high + 1

        self.dim = (dim_width, dim_high)

        if color == "GRAY":
            self.color = cv2.COLOR_BGR2GRAY
        elif color == "RGB":
            self.color = cv2.COLOR_BGR2RGB

        del dim_width, dim_high

        self.log.info("Built video handling")

    def upload_video(self, filename: str) -> list:
        """Loads a video based on your location and scales it based on pixel density and aspect ratio

        :param filename: Location of the video to upload
        :type filename: str
        :return: List containing each frame of the already scaled video
        :rtype: list
        """
        self.log.debug("Starting to upload video ...")

        video = cv2.VideoCapture(filename=filename)

        list_frames = []
        while video.isOpened():
            ret, frame = video.read()

            if ret:
                frame = cv2.resize(frame, self.dim, interpolation=cv2.INTER_AREA)
                frame = cv2.cvtColor(frame, self.color)

                list_frames.append(frame)
                # cv2.imshow("video", frame)
                # cv2.waitKey(1)
            else:
                del frame, ret
                break
        else:
            self.log.error("No se encontró la ruta: {0}".format(filename))

        video.release()
        # cv2.destroyAllWindows()
        del video

        self.log.debug("Finished to Upload video ...")
        self.log.info("Uploaded video")

        return list_frames
