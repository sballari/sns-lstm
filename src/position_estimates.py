"""Module that defines the functions for the position estimation"""
import math
import tensorflow as tf


def social_train_position_estimate(cell_output, coordinates_gt, output_size, *args):
    """Calculate the parameters to sample the new coordinates in training phase.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      coordinates_gt: tensor of shape [max_num_ped, 2]. Ground truth
        coordinates.
      output_size: int. Dimension of the output size.

    Returns:
      tuple containing the value to minimize.

    """
    # Calculate the new coordinates based on Graves (2013) equations. Assume a
    # bivariate Gaussian distribution.
    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 - 22
        # Split and squeeze to have shape [max_num_ped]
        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        std_x = tf.exp(std_x)
        std_y = tf.exp(std_y)
        rho = tf.tanh(rho)

        # Equations 24 & 25
        stds = tf.multiply(std_x, std_y)
        rho_neg = tf.subtract(1.0, tf.square(rho))

        # Calculate Z
        z_num1 = tf.subtract(coordinates_gt[:, 0], mu_x)
        z_num2 = tf.subtract(coordinates_gt[:, 1], mu_y)
        z_num3 = tf.multiply(2.0, tf.multiply(rho, tf.multiply(z_num1, z_num2)))
        z = (
            tf.square(tf.div(z_num1, std_x))
            + tf.square(tf.div(z_num2, std_y))
            - tf.div(z_num3, stds)
        )

        # Calculate N
        n_num = tf.exp(tf.div(-z, 2 * rho_neg))
        n_den = tf.multiply(
            2.0, tf.multiply(math.pi, tf.multiply(stds, tf.sqrt(rho_neg)))
        )
        return tf.div(n_num, n_den), None


def social_sample_position_estimate(
    cell_output, coordinates_gt, output_size, layer_output
):
    """Calculate the coordinates in sampling phase.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      coordinates_gt: tensor of shape [max_num_ped, 2]. Ground truth
        coordinates.
      output_size: int. Dimension of the output size.
      layer_output: tf.layer instance. Layer used for process the new
        coordinates sampled.

    Returns:
      tuple containing the new coordinates sampled and the output of
        layer_output with the new coordinates

    """

    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 - 22
        # Split and squeeze to have shape [max_num_ped]
        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        std_x = tf.exp(std_x)
        std_y = tf.exp(std_y)

        # Sample the coordinates
        dist = tf.distributions.Normal([mu_x, mu_y], [std_x, std_y])
        coordinates = tf.transpose(dist.sample())
        return coordinates, layer_output(coordinates)