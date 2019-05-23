"""Module that defines the classes that provides the input for the classes
defined in the dataset module. Each class load datasets, preprocess them and
create two generators that return sequences or batches of trajectories.

"""
import os
import random
import logging
import numpy as np
import tensorflow as tf
from multiprocessing import Pool


class DataLoader:
    """Data loader class that load the given datasets, preprocess them and create
    two generators that return sequences or batches of trajectories.

    """

    def __init__(
        self,
        data_path,
        datasets,
        trajectory_size=20,
        delimiter="\t",
        skip=1,
        navigation_maps=None,
        navigation_mapping=None,
        neighborood_size=2,
        semantic_maps=None,
        semantic_mapping=None,
        homography=None,
        num_labels=6,
    ):
        """Constructor of the DataLoader class.

        Args:
          data_path: string. Path to the folder containing the datasets
          datasets: list. List of datasets to use.
          trajectories_size: int. Length of the trajectory (obs_length +
            pred_len).
          delimiter: string. Delimiter used to separate data inside the
            datasets.
          skip: int or True. If True, the number of frames to skip while making
            the dataset is random. If int, number of frames to skip while making
            the dataset
          navigation_maps: list. List of the navigation map.
          navigation_mapping: list. Mapping between navigation_maps and
            datasets.
          neighborood_size: int. Neighborhood size.
          semantic_maps: list. List of the semantic map.
          semantic_mapping: list. Mapping between semantic_maps and datasets.
          homography: list. List of homography matrix.
          num_labels: int. Number of labels inside the semantic map.

        """
        # Store the list of datasets to load
        self.__datasets = [os.path.join(data_path, dataset) for dataset in datasets]
        logging.debug(
            "Number of dataset will be loaded: {} List of datasets: {}".format(
                len(self.__datasets), self.__datasets
            )
        )

        # Store trajectory_size, the skip value and the delimiter
        self.trajectory_size = trajectory_size
        self.skip = skip

        if delimiter == "tab":
            delimiter = "\t"
        elif delimiter == "space":
            delimiter = " "

        # If not None, store the list of the navigation maps and the neighborood
        # size
        if navigation_maps:
            self.__navigation = [
                os.path.join(data_path, navigation) for navigation in navigation_maps
            ]
            self.neighborood_size = neighborood_size

        # If not None, store the list of the semantic maps, homography and the
        # number of labels
        if semantic_maps:
            self.__semantic = [
                os.path.join(data_path, semantic) for semantic in semantic_maps
            ]
            self.__homography = [os.path.join(data_path, hg) for hg in homography]
            self.num_labels = num_labels

        # Load the datasets and preprocess them
        self.__load_data(delimiter, navigation_mapping, semantic_mapping)
        self.__preprocess_data()
        self.__type_and_shape()

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

    def __load_data(self, delimiter, navigation_mapping, semantic_mapping):
        """Load the datasets and define the lists __frames, __navigation_map,
        __top_left, __semantic_map, __homography_matrix.

        Load the datasets required by the model. __frames has shape
        [num_datasets, num_frames_dataset, num_peds_frame, 4] where 4 is
        frameID, pedID, x and y. If navigation_mapping is not None, it defines
        __navigation_map with shape [num_datasets, navigatioHeight,
        navigationWidth] and top_left with shape[num_datasets, 2]. If
        semantic_mapping is not None, it defines __semantic_map with shape
        [num_datasets, imageHeight, imageWidth, numLabels] and
        __homography_matrix with shape [num_datasets, 3, 3]

        Args:
          delimiter: string. Delimiter used to separate data insid1e the
            datasets.
          navigation_mapping: list. Mapping between navigation_maps and
            datasets.
          semantic_mapping: list. Mapping between semantic_maps and datasets.

        """
        # List that contains all the frames of the datasets. Each dataset is a
        # list of frames of shape (num_peds, (frameID, pedID, x and y))
        self.__frames = []
        self.__navigation_map = []
        self.__top_left = []
        self.__semantic_map = []
        self.__homography_matrix = []

        def load_semantic_map(smap, hom_path):
            # Load the semantic map. It is a numpy array of shape [imageHeight,
            # imageWidth, numLabels]
            semantic_map = np.load(smap)
            # Load the homography matrix. The model requires to convert the
            # image coordinates in world coordinates
            homography = np.loadtxt(hom_path, delimiter=delimiter)
            filename = os.path.splitext(os.path.basename(smap))[0]
            return (filename, (load_semantic_map, homography))

        def load_datasets(dataset):
            # Load the dataset. Each line is formed by frameID, pedID, x, y
            dataset = np.loadtxt(dataset_path, delimiter=delimiter)
            # Get the frames in dataset
            num_frames = np.unique(dataset[:, 0])
            # For each frame add to frames the pedestrians that appear in the
            # current frame
            frames = map(lambda x: dataset[dataset[:, 0] == x], num_frames)
            return frames

        def load_navigation(nmap):
            # Load the navigation map. It is a numpy array of shape
            navigation_map = np.load(nmap)
            # Image has padding so we add padding to top_left
            top_left = [
                np.floor(min(dataset[:, 2]) - self.neighborood_size / 2),
                np.ceil(max(dataset[:, 3]) + self.neighborood_size / 2),
            ]
            filename = os.path.splitext(os.path.basename[nmap])[0]
            return (filename, (navigation_map, top_left))

        # Process datasets using multiprocessing
        with Pool() as p:
            dataset_results = p.starmap_async(load_datasets, self.__datasets)
            if navigation_mapping is not None:
                navigation_results = p.starmap_async(
                    load_semantic_map, self.__navigation
                )
            if semantic_mapping is not None:
                semantic_results = p.starmap_async(
                    load_semantic_map, zip(self.__semantic, self.__homography)
                )

        # Store the results
        self.__frames = dataset_results

        if navigation_mapping is not None:
            navigation_dict = dict(navigation_results)
            for i in range(self.__datasets):
                values = semantic_dict[navigation_mapping[i]]
                self.__navigation_map.append(values[0])
                self.__top_left.append(values[1])

        if semantic_mapping is not None:
            semantic_dict = dict(semantic_results)
            for i in range(self.__datasets):
                values = semantic_dict[semantic_mapping[i]]
                self.__semantic_map.append(values[0])
                self.__homography_matrix.append(values[1])

    def __preprocess_data(self):
        """Preprocess the datasets and define the number of sequences and batches.

        The method iterates on __frames saving on the list __trajectories only
        the trajectories with length trajectory_size.

        """
        # Keep only the trajectories trajectory_size long
        self.__trajectories = []
        self.__num_peds = []
        self.num_sequences = 0

        def process_frames(dataset):
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

            return (trajectories, num_peds)

        with Pool() as p:
            results = p.starmap_async(process_frames, self.__frames)

        for result in results:
            self.__trajectories.append(results[0])
            self.__num_peds.append(results[1])

        logging.info("There are {} sequences in loader".format(self.num_sequences))

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
            [peds_sequence, trajectory_size, 4].

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
