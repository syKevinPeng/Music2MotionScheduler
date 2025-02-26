import labelbox
import numpy as np


def process_annotation_json(data):
    # Extract frame count from media attributes
    frame_count = data['media_attributes']['frame_count']
    # Initialize a 2D array with frame_count
    result = np.empty((2, frame_count), dtype=object)

    # Navigate to the annotations section
    project_id = next(iter(data['projects']))  # Assuming there's only one project
    labels = data['projects'][project_id]['labels']
    if not labels:
        return result
    annotations = labels[0]['annotations']
    
    # Extract segments and key_frame_feature_map
    segments = annotations.get('segments', {})
    key_frame_feature_map = annotations.get('key_frame_feature_map', {})
    frames_data = annotations.get('frames', {})

    # Create a mapping from feature_id to move name
    feature_to_move = {}
    for feature_id in segments:
        # Get the first frame in the key_frame_feature_map for this feature_id
        if feature_id not in key_frame_feature_map or not key_frame_feature_map[feature_id]:
            raise ValueError(f'No key frames found for feature_id: {feature_id}')
        first_frame = key_frame_feature_map[feature_id][0]
        frame_str = str(first_frame)
        if frame_str not in frames_data:
            raise ValueError(f'Frame data not found for frame: {frame_str}')
        # Extract the move name from the radio_answer
        classifications = frames_data[frame_str].get('classifications', [])
        if not classifications:
            raise ValueError(f'No classifications found for frame: {frame_str}')
        radio_answer = classifications[0].get('radio_answer', {})
        move_name = radio_answer.get('name')
        category_name = classifications[0]['name']
        if move_name:
            feature_to_move[feature_id] = (category_name,move_name)

    # Update the result array based on segments
    for feature_id, ranges in segments.items():
        category_name, move_name = feature_to_move.get(feature_id)
        if not move_name:
            raise ValueError(f'No move name found for feature {feature_id}')
        for range_ in ranges:
            start, end = range_
            # Ensure the range is within frame_count
            start = max(0, start)
            end = min(end, frame_count - 1)
            for frame in range(start, end + 1):
                result[1][frame] = move_name
                result[0][frame] = category_name

    return result


if __name__ == '__main__':
  from labelbox_key import LB_API_KEY, task_id

  client = labelbox.Client(api_key = LB_API_KEY)
  export_task = labelbox.ExportTask.get_task(client, task_id)

  # Stream the export using a callback function
  def json_stream_handler(output: labelbox.BufferedJsonConverterOutput):
    print(output.json)

  export_task.get_buffered_stream(stream_type=labelbox.StreamType.RESULT).start(stream_handler=json_stream_handler)

  # Simplified usage
  export_json = [data_row.json for data_row in export_task.get_buffered_stream()]