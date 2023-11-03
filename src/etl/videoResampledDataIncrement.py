import numpy as np

from utils.logger import Logger


class VideoResampledDataIncrement:
    """Functional used to resample the fps rate of a video and apply data increment transformation"""

    def __init__(self, video_length: int = 12) -> None:
        """Builder

        :param video_length: New video length, defaults to 12
        :type video_length: int, optional
        """
        # settings
        logger = Logger()
        self.log = logger.config_logging()

        self.video_length = video_length

        self.log.info("Built video resample data increment")

    def videoResampled(self, frames: list) -> np.ndarray:
        """It allows you to take a list of phrases and obtain a new list of frames of defined length.
        The new list of frames is obtained by subdividing the total length of the original list into
        the n required length and taking a sample using a uniform distribution.

        :param frames: List containing the frames of a video
        :type frames: list
        :return: New list made up of a sample of the original frames.
        :rtype: np.ndarray
        """
        self.log.debug("Starting video resample ...")

        current_length = len(frames)
        dividing_units = int(current_length / self.video_length)

        count_units = 0
        new_frames = []

        while count_units < self.video_length:
            index = int(np.random.randint(dividing_units, size=1))
            new_frames.append(frames[index])

            del frames[0:dividing_units]
            count_units += 1

        del current_length, dividing_units, count_units, frames

        self.log.debug("Finished resample video")
        self.log.info("Resampling video completed")

        return np.array(new_frames)
