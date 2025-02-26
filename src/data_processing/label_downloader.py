import labelbox
import numpy as np


class LabelDownloader:
    def __init__(self, LB_API_KEY, update_ontology_cache=False):
        self.client =  labelbox.Client(api_key = LB_API_KEY)
        # check if class_list.txt exists
        if not Path('class_list.txt').exists() or update_ontology_cache:
            self.class_list = self.get_ontology()
        else:
            with open('class_list.txt', 'r') as f:
                self.class_list = [line.strip().split(',') for line in f]


    def process_annotation_json(self, data):
        file_name = data["data_row"]["external_id"]
        # Extract frame count from media attributes
        frame_count = data['media_attributes']['frame_count']
        # Initialize a 2D array with frame_count
        result = np.empty((2, frame_count), dtype=object)

        assert len(data['projects'].values()) == 1
        assert len(next(iter(data['projects'].values()))['labels']) == 1


        frames=next(iter(data['projects'].values()))['labels'][0]['annotations']['frames']
        frames = [(int(k),v) for k,v in frames.items()]
        frames2=[]
        for fr,x in frames:    
            classis=x['classifications']
            for classi in classis:
                classis2 = classi['checklist_answers'] if 'checklist_answers' in classi else [classi['radio_answer']]
                for classi2 in classis2:                    
                    frames2.append((int(fr), classi['name'],classi2['name']))
        frames2.sort()

        # iterate over the frames and fill the array
        for i in range(len(frames2)):
            current_entry = frames2[i]
            start_frame = current_entry[0]
            class_label = current_entry[1]
            subclass_label = current_entry[2]
            
            # Determine the end frame based on the next entry or the total frame count
            if i < len(frames2) - 1:
                next_frame = frames2[i + 1][0]
                end_frame = next_frame - 1
            else:
                end_frame = frame_count  # Last entry ends at the maximum frame
            
            # Check if the start frame is valid
            if start_frame > end_frame:
                continue
            
            # Convert frame numbers to 0-based indices for slicing
            start_idx = start_frame - 1
            end_idx = end_frame
            
            # Assign the labels to the corresponding frames
            result[0][start_idx:end_idx] = class_label
            result[1][start_idx:end_idx] = subclass_label

        return file_name, result

    def load_labeling(self, task_id):
        export_task = labelbox.ExportTask.get_task(self.client, task_id)

        # Stream the export using a callback function
        def json_stream_handler(output: labelbox.BufferedJsonConverterOutput):
            (output.json)

        export_task.get_buffered_stream(stream_type=labelbox.StreamType.RESULT).start(stream_handler=json_stream_handler)

        # Simplified usage
        export_json = [data_row.json for data_row in export_task.get_buffered_stream()]

        # Process the annotation JSON
        all_label = {}
        for my_data in export_json:
            file_name, result = self.process_annotation_json(my_data)
            # print(f'file name:{file_name}, result:{result}')
            all_label[file_name] = result

        return all_label

    def get_ontology(self, ontology_id = "cm6pbxeeu00o507wrcwf39bq0"):
        # get a list of all labels
        ontology = self.client.get_ontology(ontology_id)
        classifications = ontology.classifications()
        class_list = []

        for classification in classifications:
            class_name = classification.name
            sub_class_name = [option.label for option in classification.options]
            for sub_name in sub_class_name:
                class_list.append([class_name, sub_name])
        # sort the class_name first and then the sub_class_name
        class_list.sort(key=lambda x: (x[0], x[1]))
        # save as txt
        with open('class_list.txt', 'w') as f:
            for item in class_list:
                f.write("%s\n" % item)
        return class_list
    
    def class_name_to_id(self, class_name):
        for i, class_list in enumerate(self.class_list):
            if class_name == class_list[1]:
                return i
        return None
    
    def class_id_to_name(self, class_id):
        return self.class_list[class_id]

    def get_class_list(self):
        return self.class_list
    




if __name__ == '__main__':
    import sys
    sys.path.append('../../../')
    from Music2MotionScheduler.src.data_processing.labelbox_key import LB_API_KEY, task_id
    from pathlib import Path
    import pickle

    output_dir = Path("/fs/nexus-projects/PhysicsFall/data/motorica_dance_dataset")
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Initialize the LabelDownloader
    downloader = LabelDownloader(LB_API_KEY)
    all_label = downloader.load_labeling(task_id)
   
    # Save the processed labels to a file
    with open(output_dir / 'all_label.pkl', 'wb') as f:
        pickle.dump(all_label, f)

    