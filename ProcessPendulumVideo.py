import numpy as np
import cv2

from matplotlib import pyplot as plt



#Routine to fix RGB
def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))



def get_median_frame(video_stream, sample_size = 30):
    """
    Method to get a median frame used for subtraction from frames to get moving parts.

    Argument: a video file name.

    Returns the gray median frame.
    """
    #open the video_stream
    #video_stream = cv2.VideoCapture(video_name)
    # Randomly select samepe_size frames
    frame_ids = video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=sample_size)
    # Store selected frames in an array
    frames = []
    for fid in frame_ids:
        video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = video_stream.read()
        frames.append(frame)
    # Calculate the median along the time axis
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    if (debug == True):
        plt.imshow(fixColor(median_frame))
        plt.show()
    # Calculate average of the selected frames, not as usefull as median, just for testing
    #avgFrame = np.average(frames, axis=0).astype(dtype=np.uint8)
    #if (debug == True):
        #plt.imshow(fixColor(avgFrame))
        #plt.show()
    #video_stream.release()
    #convert to gray
    gray_median_frame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)
    if debug:
        plt.imshow(fixColor(gray_median_frame))

    return gray_median_frame


def background_subtraction(frame, background):
    """
    Method to subtract a previous calculated background from the frame.

    Returns a gray image of the diffence.
    """
    
    gray_sample = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if debug:
        plt.imshow(fixColor(gray_sample))
    dframe = cv2.absdiff(gray_sample, background)
    return dframe


def find_contours(dframe):
    """
    Method to find all contours in a gray frame where the background has been subtracted.

    Returns an array with those contours.
    """
    #Blur the image.
    blurred = cv2.GaussianBlur(dframe, (11,11), 0)
    #if debug:
        #plt.imshow(fixColor(blurred))
        #plt.show()
    #Get a threshold image, numbers where tested for the given video.
    ret, tframe= cv2.threshold(blurred,50,255,cv2.THRESH_BINARY)
    (cnts, _) = cv2.findContours(tframe.copy(), cv2.RETR_EXTERNAL, 
                             cv2.CHAIN_APPROX_SIMPLE)
    return cnts

def find_lowest_contour(cnts):
    """
    Method to return the x,y of the contour with the largest y coordinate.
    """
    ymax = 0
    cntmax = None
    for cnt in cnts:
        x,y,w,h = cv2.boundingRect(cnt)
        if y > ymax and y < 800:
            cntmax = cnt
            ymax = y
    x,y,w,h = cv2.boundingRect(cntmax)
    x,y = x+w//2, y+h//2 #center
    return x,y

def convert_video(video_file, output_file_name):
    """
    Method takes a video file name and outputs a new video with tracked movement.
    It also returns an array of position coordinates of center of movement in each frame.
    """
    video_stream = cv2.VideoCapture(video_file)
    total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
    background = get_median_frame(video_stream)
    video_stream.release()
    #reopen for processing:
    video_stream = cv2.VideoCapture(video_file)
    #ready an output writer
    writer = cv2.VideoWriter(output_file_name, 
                        cv2.VideoWriter_fourcc(*"MP4V"), fps,(1080,1920)) #(1920,1080))
    frameCnt=0
    pos = [] #Array for the coordinates
    while(frameCnt < total_frames-1):
        frameCnt+=1
        ret, frame = video_stream.read()
        dframe = background_subtraction(frame,background)
        cnts = find_contours(dframe)
        x,y = find_lowest_contour(cnts)
        pos.append([x,y])
        if len(pos):        
            cv2.polylines(frame,np.int32([pos]),False,(0, 255, 0),2)
        writer.write(cv2.resize(frame, (1080,1920)))  ## size probably shoudn't be fixed.
    writer.release()
    video_stream.release()
    return pos

def write_data_to_file(pos, fps, data_file):
    """
    Writes the pos data ([x,y] Koordinates) and fps to file for later use.
    """
    xs = []
    for x,y in pos:
        xs.append(x)
    with open(data_file,'wb') as f:
        np.save(f,pos)
        np.save(f,xs)
        np.save(f,fps)

def analyze_fft_pendulum_data(pos, threshold, fps):
    xs = []
    for x,y in pos:
        xs.append(x)
    N = len(xs)
    W = np.fft.fft(xs)
    freq = np.fft.fftfreq(N,1)
    #threshold = 2*10**4
    idx = np.where(abs(W)>threshold)[0][-1]
    max_f = abs(freq[idx])
    period =  1/max_f/fps
    #print ("Period estimate: ", 1/max_f/fps)
    return period


debug = False

if __name__ == '__main__':
    #set True to see intermediate images
    np.random.seed(42)
    file_name = "Video/Pendel-schoen.mp4"
    #open video
    video_stream = cv2.VideoCapture(file_name)
    #get fps from video
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    output_file_name = "output.mp4"
    positions = convert_video(file_name, output_file_name)
    #close stream 
    video_stream.release()
    #writer.release()
    write_data_to_file(positions, fps, "pendulum_data.npy")
    #analyze data:
    period = analyze_fft_pendulum_data(positions,10**4,fps)
    print("The period of the oscilation is {:.2f} s".format(period))
    
