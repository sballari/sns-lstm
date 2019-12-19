"""Module that defines the classes that provides the input for the classes
defined in the dataset module. Each class load datasets, preprocess them and
 two generators that return sequences or batches of trajectories.

"""
import os
import random
import logging
import numpy as np
import tensorflow as tf


class DataLoader:
    """Data loader class that load the given datasets, preprocess them and create
    two generators that return sequences or batches of trajectories.

    """

    def __init__(
        self,
        data_path,
        datasets,
        navigation_maps,
        semantic_maps,
        semantic_mapping,
        homography,
        num_labels=6,
        delimiter="\t",
        skip=1,
        max_num_ped=100,
        trajectory_size=20,
        neighborood_size=2,
        batch_size=10,
    ):
        """Constructor of the DataLoader class.

        Args:
          data_path: string. Path to the folder containing the datasets
          datasets: list. List of datasets to use.
          navigation_maps: list. List of the navigation map.
          semantic_maps: list. List of the semantic map.
          semantic_mapping: list. Mapping between semantic_maps and datasets.
          num_labels: int. Number of labels inside the semantic map.
          homography: list. List of homography matrix.
          delimiter: string. Delimiter used to separate data inside the
            datasets.
          skip: int or True. If True, the number of frames to skip while making
            the dataset is random. If int, number of frames to skip while making
            the dataset
          max_num_ped: int. Maximum number of pedestrian in a single frame.
          trajectories_size: int. Length of the trajectory (obs_length +
            pred_len).
          neighborood_size: int. Neighborhood size.
          batch_size: int. Batch size.

        """
        # Store the list of datasets to load
        self.__datasets = [os.path.join(data_path, dataset) for dataset in datasets]
        logging.debug(
            "Number of dataset will be loaded: {} List of datasets: {}".format(
                len(self.__datasets), self.__datasets
            )
        )

        # Store the list of the navigation map
        self.__navigation = [
            os.path.join(data_path, navigation) for navigation in navigation_maps
        ]
        # Store the list of the semantic map
        self.__semantic = [
            os.path.join(data_path, semantic) for semantic in semantic_maps
        ]
        # Store the list of the homography matrix
        self.__homography = [os.path.join(data_path, hg) for hg in homography]

        # Store the batch_size, trajectory_size, the maximum number of
        # pedestrian in a single frame and skip value
        self.batch_size = batch_size
        self.trajectory_size = trajectory_size
        self.max_num_ped = max_num_ped
        self.skip = skip
        self.neighborood_size = neighborood_size
        self.num_labels = num_labels

        if delimiter == "tab":
            delimiter = "\t"
        elif delimiter == "space":
            delimiter = " "

        # Load the datasets and preprocess them
        self.__load_data(delimiter, semantic_mapping)
        self.__preprocess_data()
        self.__type_and_shape()

    def next_batch(self):
        """Generator method that returns an iterator pointing to the next batch.

        Returns:
          Generator object that has a list of trajectory sequences of size
            batch_size, a list of relative trajectory sequences of size
            batch_size, a list containing the mask for the grid layer of size
            batch_size, a list with the number of pedestrian in each sequence, a
            list containing the mask for the loss function, a list containing
            the navigation map, the top_left coordinates for each dataset and a
            list containing the semantic maps.

        """
        it = self.next_sequence()
        for batch in range(self.num_batches):
            batch = []
            batch_rel = []
            mask_batch = []
            peds_batch = []
            loss_batch = []
            navigation_map_batch = []
            top_left_batch = []
            semantic_map_batch = []
            homography_matrix = []

            for size in range(self.batch_size):
                data = next(it)
                batch.append(data[0])
                batch_rel.append(data[1])
                mask_batch.append(data[2])
                peds_batch.append(data[3])
                loss_batch.append(data[4])
                navigation_map_batch.append(data[5])
                top_left_batch.append(data[6])
                semantic_map_batch.append(data[7])
                homography_matrix.append(data[8])
            yield (
                batch,
                batch_rel,
                mask_batch,
                peds_batch,
                loss_batch,
                navigation_map_batch,
                top_left_batch,
                semantic_map_batch,
                homography_matrix,
            )

    def next_sequence(self):
        """Generator method that returns an iterator pointing to the next sequence.

        Returns:
          Generator object that contains a trajectory sequence, a relative
            trajectory sequence, the mask for the grid layer, the number of
            pedestrian in the sequence, the mask for the loss function, the
            navigation map, the top_left coordinates for the datset and the
            semantic map.

        """
        # Iterate through all sequences
        for idx_d, dataset in enumerate(self.__trajectories):
            # Every dataset
            for idx_s, trajectories in enumerate(dataset):
                sequence, mask, loss_mask = self.__get_sequence(trajectories)

                # Create the relative coordinates
                sequence_rel = np.zeros(
                    [self.trajectory_size, self.max_num_ped, 2], float
                )
                sequence_rel[1:] = sequence[1:] - sequence[:-1]
                num_peds = self.__num_peds[idx_d][idx_s]

                yield (
                    sequence,
                    sequence_rel,
                    mask,
                    num_peds,
                    loss_mask,
                    self.__navigation_map[idx_d],
                    self.__top_left[idx_d],
                    self.__semantic_map[idx_d],
                    self.__homography_matrix[idx_d],
                )

    def __load_data(self, delimiter, semantic_mapping):
        """Load the datasets and define the list __frames.

        Load the datasets and define the list __frames wich contains all the
        frames of the datasets and the list __navigation_map. __frames has shape
        [num_datasets, num_frames_dataset, num_peds_frame, 4] where 4 is
        frameID, pedID, x and y.

        Args:
          delimiter: string. Delimiter used to separate data inside the
            datasets.

        """
        # List that contains all the frames of the datasets. Each dataset is a
        # list of frames of shape (num_peds, (frameID, pedID, x and y))
        self.__frames = []
        self.__navigation_map = []
        self.__top_left = []
        self.__semantic_map = []
        self.__homography_matrix = []
        semantic_map_labeled = {}
        homography_map = {}

        # Load and add the one hot encoding to the semantic maps
        for i, smap in enumerate(self.__semantic):
            # Load the semantic map
            semantic_map = np.load(smap)
            homography = np.loadtxt(self.__homography[i], delimiter=delimiter)
            filename = os.path.splitext(os.path.basename(smap))[0]
            semantic_map_labeled[filename] = semantic_map
            homography_map[filename] = homography

        for i, dataset_path in enumerate(self.__datasets):
            # Load the dataset. Each line is formed by frameID, pedID, x, y
            dataset = np.loadtxt(dataset_path, delimiter=delimiter)
            # Get the frames in dataset
            num_frames = np.unique(dataset[:, 0])
            # Initialize the array of frames for the current dataset
            frames_dataset = []
            # Load the navigation map
            navigation_map = np.load(self.__navigation[i])

            # Image has padding so we add padding to the top_left point.
            top_left = [
                np.floor(min(dataset[:, 2]) - self.neighborood_size / 2),
                np.ceil(max(dataset[:, 3]) + self.neighborood_size / 2),
            ]

            # For each frame add to frames_dataset the pedestrian that appears
            # in the current frame
            for frame in num_frames:
                # Get the pedestrians
                frame = dataset[dataset[:, 0] == frame, :]
                frames_dataset.append(frame)

            self.__frames.append(frames_dataset)
            self.__navigation_map.append(navigation_map)
            self.__top_left.append(top_left)
            self.__semantic_map.append(semantic_map_labeled[semantic_mapping[i]])
            self.__homography_matrix.append(homography_map[semantic_mapping[i]])

    def __preprocess_data(self):
        """Preprocess the datasets and define the number of sequences and batches.

        The method iterates on __frames saving on the list __trajectories only
        the trajectories with length trajectory_size.

        """
        # Keep only the trajectories trajectory_size long
        self.__trajectories = []
        self.__num_peds = []
        self.num_sequences = 0

        for dataset in self.__frames:
            # Initialize the array of trajectories for the current dataset.
            trajectories = []
            num_peds = []
            frame_size = len(dataset)
            i = 0

            # Each trajectory contains only frames of a dataset
            while i + self.trajectory_size < frame_size:
                sequence = dataset[i : i + self.trajectory_size]
                # Get the pedestrians in the first frame
                peds = np.unique(sequence[0][:, 1])
                # Check if the trajectory of pedestrian is long enough.
                sequence = np.concatenate(sequence, axis=0)
                traj_frame = []
                for ped in peds:
                    # Get the frames where ped appear
                    frames = sequence[sequence[:, 1] == ped]
                    # Check the trajectory is long enough
                    if frames.shape[0] == self.trajectory_size:
                        traj_frame.append(frames)
                # If no trajectory is long enough traj_frame is empty. Otherwise
                if traj_frame:
                    trajectories_frame, peds_frame = self.__create_sequence(
                        traj_frame, sequence
                    )
                    trajectories.append(trajectories_frame)
                    num_peds.append(peds_frame)
                    self.num_sequences += 1
                # If skip is True, update the index with a random value
                if self.skip is True:
                    i += random.randint(0, self.trajectory_size)
                else:
                    i += self.skip

            self.__trajectories.append(trajectories)
            self.__num_peds.append(num_peds)

        # num_batches counts only full batches. It discards the remaining
        # sequences
        self.num_batches = int(self.num_sequences / self.batch_size)
        logging.info("There are {} sequences in loader".format(self.num_sequences))
        logging.info("There are {} batches in loader".format(self.num_batches))

    def __get_sequence(self, trajectories):
        """Returns a tuple containing a trajectory sequence, the mask for the grid layer
        and the mask for the loss function.

        Args:
          trajectories: list of numpy array. Each array is a trajectory.

        Returns:
          tuple containing a numpy array with shape [trajectory_size,
            max_num_ped, 2] that contains the trajectories, a numpy array with
            shape [trajectory_size, max_num_ped, max_num_ped] that is the mask
            for the grid layer and a numpy array with shape [trajectory_size,
            max_num_ped] that is the mask for the loss function.

        """
        num_peds_sequence = len(trajectories)
        sequence = np.zeros((self.max_num_ped, self.trajectory_size, 2))
        mask = np.zeros((self.max_num_ped, self.trajectory_size), dtype=bool)

        sequence[:num_peds_sequence] = trajectories[:, :, [2, 3]]

        # Create the mask for the grid layer. Set to True only the pedestrians
        # that are in the sequence. A pedestrian is in the sequence if its
        # frameID is not 0
        mask[:num_peds_sequence] = trajectories[:, :, 0]
        # Create the mask for the loss function
        loss_mask = mask
        # Create the mask for all the pedestrians
        mask = np.tile(mask, (self.max_num_ped, 1, 1))
        # The mask ignores the pedestrian itself
        for ped in range(num_peds_sequence):
            mask[ped, ped] = False

        # Change shape of the arrays. From [max_num_ped, trajectory_size] to
        # [trajectory_size, max_num_ped]
        sequence_moved = np.moveaxis(sequence, 1, 0)
        mask_moved = np.moveaxis(mask, 2, 0)
        loss_moved = np.moveaxis(loss_mask, 1, 0)

        return sequence_moved, mask_moved, loss_moved

    def __create_sequence(self, trajectories_full, sequence):
        """Create an array with the trajectories contained in a dataset slice.

        Args:
          trajectories_full: list that contains the trajectories long
            trajectory_size of the dataset slice.
          sequence: list that contains the remaining trajectories of the dataset
            slice.

        Returns:
          tuple containing the ndarray with the trajectories of the dataset
            slice and the number of pedestrians thate are trajectory_size long
            in the dataset slice. In the first positions of the ndarray there
            are the trajectories long enough. The shape of the ndarray is
            [peds_sequence, trajectory_size, 4]. (x,y,frame_id,id_pedone)

        """
        trajectories_full = np.array(trajectories_full)
        peds_sequence = np.unique(sequence[:, 1])
        peds_trajectories = np.unique(trajectories_full[:, :, 1])
        frames_id = np.unique(sequence[:, 0])
        # Create the array that will contain the trajectories
        trajectories = np.zeros((len(peds_sequence), self.trajectory_size, 4))

        # Copy trajectories_full in the first len(peds_trajectories) rows
        trajectories[: len(peds_trajectories)] = trajectories_full
        # Remove the peds that are in peds_trajectories
        peds_sequence = np.delete(
            peds_sequence, np.searchsorted(peds_sequence, peds_trajectories)
        )
        # Create a lookup table with the frames id and their position in the
        # sequence
        lookup_frames = {}
        for i, frame in enumerate(frames_id):
            lookup_frames[frame] = i

        # Add the remaining peds
        for i, ped in enumerate(peds_sequence, len(peds_trajectories)):
            # Get the indexes where the pedsID is equal to ped
            positions = np.where(sequence[:, 1] == ped)[0]
            # Use the lookup table to find out where the pedestrian trajectory
            # begins and end in the sequence
            start = lookup_frames[sequence[positions][0, 0]]
            end = lookup_frames[sequence[positions][-1, 0]] + 1
            # Copy the pedestrian trajectory inside the sequence
            trajectories[i, start:end] = sequence[positions]

        return trajectories, len(peds_trajectories)

    def __type_and_shape(self):
        """Define the type and the shape of the arrays that tensorflow will use."""
        navigation_h, navigation_w = self.__navigation_map[0].shape
        self.output_types = (
            tf.float32,
            tf.float32,
            tf.bool,
            tf.int32,
            tf.int32,
            tf.float32,
            tf.float32,
            tf.float32,
            tf.float32,
        )
        self.shape = (
            tf.TensorShape([self.trajectory_size, self.max_num_ped, 2]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped, 2]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped, self.max_num_ped]),
            tf.TensorShape([]),
            tf.TensorShape([self.trajectory_size, self.max_num_ped]),
            tf.TensorShape([navigation_h, navigation_w]),
            tf.TensorShape([2]),
            tf.TensorShape([None, None, self.num_labels]),
            tf.TensorShape([3, 3]),
        )
