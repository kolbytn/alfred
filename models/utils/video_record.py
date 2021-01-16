# video recorder helper class
from datetime import datetime
import cv2
class VideoRecord:
    def __init__(self, path, name, fps=5):
        """
        param:
            path: video save path (str)
            name: video name (str)
            fps: frames per second (int) (default=5)
        example usage:
            rec = VideoRecord('path/to/', 'filename', 10)
        """
        self.path = path
        self.name = name
        self.fps = fps
        self.frames = []
    def record_frame(self, env_frame):
        """
            records video frame in this object
        param:
            env_frame: a frame from thor environment (ThorEnv().last_event.frame)
        example usage:
            env = Thorenv()
            lastframe = env.last_event.frame
            rec.record_frame(lastframes)
        """
        curr_image = Image.fromarray(np.uint8(env_frame))
        img = cv2.cvtColor(np.asarray(curr_image), cv2.COLOR_RGB2BGR)
        self.frames.append(img)
    def savemp4(self):
        """
            writes video to file at specified location, finalize video file
        example usage:
            rec.savemp4()
        """
        if len(self.frames) == 0:
            raise Exception("Can't write video file with no frames recorded")
        height, width, layers = self.frames[0].shape
        size = (width,height)
        out = cv2.VideoWriter(f"{self.path}{self.name}.mp4", 0x7634706d, self.fps, size)
        for i in range(len(self.frames)):
            out.write(self.frames[i])
        out.release()