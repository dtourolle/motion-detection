import numpy as np
from scipy.ndimage.morphology import binary_closing , binary_erosion,binary_opening


class EMAV_background_subtracting_motion_detection:
    def __init__(self, weight, threshold, shape, initial_average=0, initial_variance=0, minimum_variance=0,closing_iteration=2):
        """
        EMA_background_subtracting_motion_detection(weight, threshold, shape)

        A class which detects motion based upon a running average and variance
        of a scene. The running average and variance ar calculated using a 
        exponential running average filter. The average value is subtracted from
        new frames to determine the square difference, while this is divided by 
        the variance to determine how likely the update value ocurred naturally
        within a scene.
        Important to note that the variance can be comprised of more than just
        sensor noise, also regularly occuring movements in the background such
        as plants moving will increase the variance.

        Parameters
        ----------
        weight : float or ndarray
        The weighting factor for the background learning, must be <=1.0. Can be an
        ndnumpy array for scenes in which backgrounds have require different learning
        rates e.g. Sky versus ground, or different colour channels.
        threshold : float
        The threshold for determining if an a change motion of naturally occuring in
        the background. If you know the distribution of noise in your background and
        it's variance then you can calculate a good value a priori e.g. Normal 
        distributed a `threshold` of 3 will exclude 99.87% of all natural noise, this
        may seem small but on a RGB 2MP image that is still falsely detecting ~8000 
        pixels as motion each update.
        shape: tuple
        the shape of the image buffer
        initial_average : float/int/ndarray
        The initial value of the background, will speed up learning if close to correct
        value.
        initial_variance : float/int/ndarray
        The initial variance of the background, will speed up learning if close to correct
        value.
        minimum_variance : float/int/ndarray
        It can be useful to limmit the lower level of variance, images from cameras are 
        often clipped in high intensity areas or clamped in low, which reduces the
        variance. The motion of background objects might be too slow to maintain a high
        variance, e.g. shadows moving with the sun, a fixed minimum threshold can reduce
        false detections.
        closing_iteration : int
        number of iterations of closing algorithm to connect detected pixels


        """
        self.weight = weight
        
        self.threshold=threshold
        self.average_background = np.ones(shape)*initial_average
        self.variance_background = np.ones(shape)*initial_variance
        self.minimum_variance = minimum_variance
        self.shape = shape
        self.mask = np.zeros(shape,dtype=bool)
        self.closing_iteration=closing_iteration


    def apply(self,image):
        """
        apply(image)

        function which updates background and returns a mask of the detected motion.

        Parameters
        ----------
        image : ndarray
        The latest image. must be same size as `shape`.
        
        Returns
        -------
        ndarry
        A boolean mask of the motion in 2d.
        """
        self.update_background(image)
        return self.get_processed_mask()


    def update_background(self,image):
        """
        update_background(image)

        Function which updates background and does first step of motion detection.
        While it would be great to separate these for performance and resources it
        is best to do this in a single step and use the same temporary variables.

        Parameters
        ----------
        image : ndarray
        The latest image. must be same size as `shape`.

        """

        bw = image.astype(np.float32)#rgb2grey(image).astype(np.float32)
        diff = bw - self.average_background
        self.average_background += diff*self.weight
        square_diff=diff**2
        self.mask = (square_diff/self.variance_background) > self.threshold
        self.variance_background = (1.0-self.weight)*(self.variance_background + self.weight*square_diff)
        if self.minimum_variance != 0:
            self.variance_background[self.variance_background<self.minimum_variance]=self.minimum_variance

    def get_mask(self):
        """
        get_mask(image)

        convenience function for getting mask prior to processing and dimension reduction. This
        is best used with a custom post processing step.

        Returns
        -------
        ndarry
        A boolean mask of same shape as input image.
        """
        return self.mask

    def get_processed_mask(self):
        """
        get_mask(image)

        convenience function for getting mask which is 2d, the mask is post-processed with a
        closing operation filling in any holes in detected objects.

        Returns
        -------
        ndarry
        A boolean mask of same shape as input image.
        """

        if len(self.mask.shape)==3:
            out = binary_closing(np.sum(self.mask,axis=2)!=0,iterations=2)*255
        else:
            out = binary_closing(self.mask,iterations=2)*255
        return out
