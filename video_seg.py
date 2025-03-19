from util import get_duration, get_segment_id
import pdb

# Video segment as a tree node
class VideoSeg:
    def __init__(self, start, end, segment_id=None, description=None):
        """
        Initialize VideoSeg object
        
        :param start: int, start frame number of the video segment
        :param end: int, end frame number of the video segment
        :param segment_id: int, ID of the video segment marked by LLM, dummy
        :param description: str or None, description of the video segment (default is None), dummy
        """
        self.start = start            
        self.end = end                  

        self.segment_id = segment_id
        self.description = description 

    def __repr__(self):
        """
        Return a brief string representation of the video segment
        """
        return f"VideoSeg(start={self.start}, end={self.end}, segment_id={self.segment_id}, description={self.description})"
    
    def __eq__(self, other):
        """
        Check if two VideoSeg instances are equal
        """
        if isinstance(other, VideoSeg):
            return self.start == other.start and self.end == other.end
        return False


def extract_videoseg_from_descriptions(descriptions):
    """
    Extract video segment instances from a list of descriptions
    
    :param descriptions: List of dictionaries, each containing 'segment_id', 'duration', 'description' or 'display_description'
    :return: List of VideoSeg instances
    """
    video_segments = []
    
    for description in descriptions:

        duration = get_duration(description)

        try:
            start, end = map(int, duration.split('-'))  # Parse 'start-end' format
        except ValueError: 
            print(f"ValueError -- extract_videoseg_from_descriptions -- duration:{duration}")
            # Duration has only one number
            start = int(duration)
            end = int(duration)
            # pdb.set_trace()
        except:  # AttributeError: 'NoneType' object(duration) has no attribute 'split'
            print(f"Error -- extract_videoseg_from_descriptions -- description: {description}")
            continue

        segment_id = get_segment_id(description)
        
        # Get description (prefer 'description', if not available set to None)
        description = description.get('description', None)
        
        # Create VideoSeg instance
        video_seg = VideoSeg(start, end, segment_id, description)

        if video_seg not in video_segments:  # Prevent duplicates
            video_segments.append(video_seg)
    
    return video_segments



def split_and_reconnect_segments(selected_video_segments, video_segments, for_seg_not_interested, num_frames):
    """
    Split and reconnect video segments based on the specified strategy
    
    :param selected_video_segments: List of selected video segments
    :param video_segments: List of all video segments
    :param for_seg_not_interested: Strategy for segments not interested ("prune", "retain", "merge")
    :param num_frames: Total number of frames in the video
    :return: List of new video segments
    """
    new_segments = []

    if for_seg_not_interested == "prune":
    
        # Split each selected video segment into two
        for segment in selected_video_segments:
            
            if segment.start >= segment.end - 1:
                # If the segment has only one or two frames, it cannot be split
                new_segments.append(segment)
            else:
                mid_point = (segment.start + segment.end) // 2  # Calculate midpoint

                # Create two new VideoSeg instances
                first_half = VideoSeg(start=segment.start, end=mid_point)
                second_half = VideoSeg(start=mid_point, end=segment.end)
                
                # Add the newly created segments to the result list
                new_segments.append(first_half)
                new_segments.append(second_half)
    
    elif for_seg_not_interested == "retain":

        for segment in video_segments:

            if segment in selected_video_segments:
                if segment.start >= segment.end - 1:
                    # If the segment has only one frame, it cannot be split
                    new_segments.append(segment)
                else:
                    mid_point = (segment.start + segment.end) // 2  # Calculate midpoint

                    # Create two new VideoSeg instances
                    first_half = VideoSeg(start=segment.start, end=mid_point)
                    second_half = VideoSeg(start=mid_point, end=segment.end)
                    
                    # Add the newly created segments to the result list
                    new_segments.append(first_half)
                    new_segments.append(second_half)
            else:
                new_segments.append(segment)

    elif for_seg_not_interested == "merge":
        
        for i, segment in enumerate(selected_video_segments):

            if i == 0:
                # Connect the initial segment
                if segment.start != 1:
                    video_start_seg = VideoSeg(start=1, end=segment.start)
                    new_segments.append(video_start_seg)

            # Merge the missing segments into a new node
            if i != 0 and segment.start != new_segments[-1].end:
                video_merged_seg = VideoSeg(start=new_segments[-1].end, end=segment.start)
                new_segments.append(video_merged_seg)

            if segment.start >= segment.end - 1:
                # If the segment has only one frame, it cannot be split
                new_segments.append(segment)
            else:
                mid_point = (segment.start + segment.end) // 2  # Calculate midpoint

                # Create two new VideoSeg instances
                first_half = VideoSeg(start=segment.start, end=mid_point)
                second_half = VideoSeg(start=mid_point, end=segment.end)
                
                # Add the newly created segments to the result list
                new_segments.append(first_half)
                new_segments.append(second_half)

            if i == len(selected_video_segments) - 1:
                # Connect the final segment
                if segment.start != 180:
                    video_start_seg = VideoSeg(start=segment.end, end=num_frames)
                    new_segments.append(video_start_seg)
            
    else:
        raise KeyError
    
    return new_segments