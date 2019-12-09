"""Module that defines the functions for the position estimation"""
import math
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

def tf_2d_normal(x, y, mux, muy, sx, sy, rho):
            '''
            Function that implements the PDF of a 2D normal distribution
            params:
            x : input x points
            y : input y points
            mux : mean of the distribution in x
            muy : mean of the distribution in y
            sx : std dev of the distribution in x
            sy : std dev of the distribution in y
            rho : Correlation factor of the distribution
            '''
            # eq 3 in the paper
            # and eq 24 & 25 in Graves (2013)
            # Calculate (x - mux) and (y-muy)
            normx = tf.subtract(x, mux)
            normy = tf.subtract(y, muy)
            # Calculate sx*sy
            sxsy = tf.multiply(sx, sy)
            # Calculate the exponential factor
            z = tf.square(tf.divide(normx, sx)) + tf.square(tf.divide(normy, sy)) - 2.0*tf.divide(tf.multiply(rho, tf.multiply(normx, normy)), sxsy)
            negRho = 1.0 - tf.square(rho)
            # Numerator
            result = tf.exp(tf.divide(-z, 2.0*negRho))
            # Normalization constant
            denom = 2.0 * math.pi * tf.multiply(sxsy, tf.sqrt(negRho))
            # Final PDF calculation
            result = tf.divide(result, denom)
            #self.result = result
            return result

def social_train_position_estimate(cell_output, output_size, coordinates_gt):
    """Calculate the probability density function in training phase.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after applying a linear layer.
      output_size: int. Dimension of the output size.
      coordinates_gt: tensor of shape [max_num_ped, 1]. Ground truth
        coordinates.

    Returns:
      tensor of shape [max_num_ped, 2] that contains the pdf.

    """
    # Calculate the probability density function on Graves (2013) equations.
    # Assume a bivariate Gaussian distribution.
    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 - 22
        # Split and squeeze to have shape [max_num_ped]


        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        
        std_x = tf.exp(std_x) #stx>=0
        std_y = tf.exp(std_y) #sty>=0
        rho = tf.tanh(rho) #normalizza?    

        # Equations 24 & 25
        stds = tf.multiply(std_x, std_y)
        rho_neg = tf.subtract(1.0, tf.square(rho))  

        # Calculate Z
        z_num1 = tf.subtract(coordinates_gt[:, 0], mu_x)
        z_num2 = tf.subtract(coordinates_gt[:, 1], mu_y)
        z_num3 = tf.multiply(2.0, tf.multiply(rho, tf.multiply(z_num1, z_num2)))
        z1 = tf.square(tf.div(z_num1, std_x))
        z2 = tf.square(tf.div(z_num2, std_y))
        z3 = tf.div(z_num3, stds)
        z = tf.subtract(tf.add(z1,z2),z3)

        # Calculate N
        n_num = tf.exp(tf.divide(-z,tf.multiply(2.0,rho_neg)))
        n_den = tf.multiply(
            2.0, tf.multiply(math.pi, tf.multiply(stds, tf.sqrt(rho_neg)))
        )
      

        normal = tf.div(n_num, n_den)

        
        
        # normal = tf_2d_normal(coordinates_gt[:, 0],coordinates_gt[:, 1],mu_x,mu_y,std_x,std_y,rho)
        return normal


def social_sample_position_estimate(cell_output, output_size):
    """Calculate the new coordinates in sampling phase.

    Args:
      cell_output: tensor of shape [max_num_ped, output_size]. The output of the
        LSTM after the linear layer.
      output_size: int. Dimension of the output size.

    Returns:
      tensor of shape [max_num_ped, 2] that contains the sampled coordinates.

    """
  
    # Calculate the new coordinates based on Graves (2013) equations. Assume a
    # bivariate Gaussian distribution.
    with tf.name_scope("Calculate_coordinates"):
        # Equations 20 - 22 from Graves
        # Split and squeeze to have shape [max_num_ped]
        mu_x, mu_y, std_x, std_y, rho = list(
            map(lambda x: tf.squeeze(x, 1), tf.split(cell_output, output_size, 1))
        )
        std_x = tf.exp(std_x)
        std_y = tf.exp(std_y)
        rho = tf.tanh(rho)
        
        # Kaiser-Dickman algorithm (Kaiser & Dickman, 1962)
        # Generate two sample X1, X2 from the standard normal distribution
        # (mu = 0, sigma = 1)
        normal_coords = tf.random.normal(tf.TensorShape([mu_x.shape[0], 2]))
        # Generate the correlation.
        # correlation = rho * X1 + sqrt(1 - pow(rho)) * X2
        correlation = (
            rho * normal_coords[:, 0]
            + tf.sqrt(1 - tf.square(rho)) * normal_coords[:, 1]
        )

        # Define the two coordinates correlated
        # Y1 = mu_x + sigma_x * X1
        # Y2 = mu_y + sigma_y * correlation
        coords_x = mu_x + std_x * normal_coords[:, 0]
        coords_y = mu_y + std_y * correlation

        coordinates = tf.stack([coords_x, coords_y], 1)
        
        return coordinates


