# Copyright (c) 2011, Alex Krizhevsky (akrizhevsky@gmail.com)
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# 
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
# EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

############################## DIMENSIONS INFO #########################################
# data_dic is entire dataset: a list of [datadic['data'], datadic['labels']] elements. #        
#          LabeledMem only.                                                            #
# datadic['data'] is a num_colors x batchSize x img_size x img_size nparray.           #
# datadic['labels'] is a batchSize x img_size x img_size nparray?                      #
# AugmentLeaf.cropped_data is a (img_size*img_size*3)x(batchSize*datamult) nparray.    #
# CroppedCIFAR.cropped_data is a list of 2 (img_size*img_size*3)x(batchSize*datamult)  #
#                           nparrays.                                                  #
# CroppedCIFAR.cropped is a (img_size*img_size*3)x(batchSize*datamult) nparray.        #
########################################################################################

from data import *
import numpy.random as nr
import numpy as n
import random as r
from joblib import Parallel, delayed


# def augment_image(image, random_gaussian):
#     ''' Image here is a single row numpy array, with all of the channels in sequential
#     order (i.e. goes column by column then row by row for red, then same for green
#     then same for blue). '''
#     eigenvalue, eigenvector = n.linalg.eig(n.cov(image))
#     addition = n.dot(eigenvector,n.sqrt(eigenvalue).T * random_gaussian)
#     return n.clip(n.add(image.T,addition),0,255).reshape(-1)


# def augment_illumination(data,channels=3):
#     ''' Data here is a 2D matrix, with columns being images, and rows being pixels
#     of those images, in order of columns, then rows, starting with R, the G and
#     B in sequential order.  A matrix of similar shape is returned with the data
#     augmented to have different illumination levels via a PCA analysis & manipulation.'''
#     data = data.reshape(-1,channels,data.shape[1]).T
#     random_gaussian = n.random.normal(0,0.6,channels)
#     rows = Parallel(n_jobs=-1)(
#                     delayed(augment_image)(image, random_gaussian) 
#                     for image in data)
#     return n.vstack([r for r in rows]).T


# # MODIFY NOCCN?
# # Note command line crop_border arg expected! (and if --test-only provided, --multiview-test is optional)
# # Might be better to have inner_size as command line arg in case 224x224 patches are too low def for fine grained classification
# # (image resizing (eg to get all 256x256) is done in nocnn/dataset.py)

# class AugmentLeafDataProvider(LabeledDataProvider):
#     def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params={'crop_border': 16, 'multiview_test': False}, test=False):
#         LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
#         if batch_range == None:
#             batch_range = DataProvider.get_batch_nums(data_dir)
#         self.data_mean = self.batch_meta['data_mean']
#         self.num_colors = 3
#         # patch_idx: x coordinate of patch, y coordinate of patch, flip image no/yes
#         self.patch_idx = [0,0,0]
#         self.inner_size = 224
#         # border_size: such that central patch edge is border_size pixels away from original img edge (expect 16)
#         self.border_size = dp_params['crop_border'] 
#         # multiview: to compute test error averaged over top left, bottom left, central, top right, bottom right patches (use all info in img)
#         self.multiview = dp_params['multiview_test'] and test 
#         self.num_views = 5*2
#         # data_mult: multiply data matrix dimensions if in multiview test mode
#         self.data_mult = self.num_views if self.multiview else 1
#         self.data_mean = self.batch_meta['data_mean'].reshape((3,256,256))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

        
#     def get_next_batch(self):
#         if self.data_dic is None or len(self.batch_range) > 1:
#             self.data_dic = self.get_batch(self.curr_batchnum)
#         # print 'just got a batch. data shape is %s, labels shape is %s' % (self.data_dic['data'].shape, self.data_dic['labels'].shape)
#         epoch, batchnum = self.curr_epoch, self.curr_batchnum
#         cropped = self.crop_batch()
#         self.advance_batch()
#         # Subtract the mean from the data and make sure that both data and
#         # labels are in single-precision floating point.
#         # This converts the data matrix to single precision and makes sure that it is C-ordered
#         cropped = n.require((cropped - self.data_mean), dtype=n.single, requirements='C')
#         self.data_dic['labels'] = n.require(self.data_dic['labels'].reshape((1,cropped.shape[1])), dtype=n.single, requirements='C')
        
#         return epoch, batchnum, [cropped, self.data_dic['labels']]


#     # Returns the dimensionality of the two data matrices returned by get_next_batch
#     # idx is the index of the matrix.
#     def get_data_dims(self, idx=0):
#         return self.inner_size**2 * 3 if idx == 0 else 1

#     # Takes as input an array returned by get_next_batch
#     # Returns a (numCases, patchSize, patchSize, 3) array which can be
#     # fed to pylab for plotting.
#     # This is used by shownet.py to plot test case predictions.
#     def get_plottable_data(self, data):
#         return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
#     # override advance_batch to only increment epoch once data augmentation fully complete
#     def advance_batch(self):
#         self.batch_idx = self.get_next_batch_idx()
#         self.curr_batchnum = self.batch_range[self.batch_idx]
#         if self.batch_idx == 0 and self.patch_idx == [0,0,0]: # patch_idx needs to be back at 0,0,0 too
#             self.curr_epoch += 1    

#     def crop_batch(self):
#         # print 'initialising cropped to a (%i, %i)-dimensional array' % (self.get_data_dims(), self.data_dic['data'].shape[1])
#         cropped = n.zeros((self.get_data_dims(), 
#                            self.data_dic['data'].shape[1]*self.data_mult), # batch['data'].shape[1] == batchSize
#                           dtype=n.single)
#         self.__select_patch(self.data_dic['data'], self.inner_size, cropped)
#         return cropped

#     # called as __select_patch(datadic['data'], cropped)
#     def __select_patch(self, x, patch_dimension, target):
#         y = x.reshape(3, 256, 256, x.shape[1])
#         if self.test: # if --test-only=1 provided at command line
#             if self.multiview: # if --multiview-test=1 also provided at command line
#             # compute error by averaging over top left, bottom left, central, top right, bottom right patches
#                 start_positions = [(0,0),  (0, self.border_size*2),
#                                    (self.border_size, self.border_size),
#                                    (self.border_size*2, 0),
#                                    (self.border_size*2, self.border_size*2)]
#                 end_positions = [(sy+self.inner_size, sx+self.inner_size)
#                                  for (sy,sx) in start_positions]
#                 for i in xrange(self.num_views/2):
#                     pic = y[:,start_positions[i][0]:end_positions[i][0],
#                             start_positions[i][1]:end_positions[i][1],:]
#                     target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
#                     target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
#             else:
#                 # select only central patch in image
#                 pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :]
#                 target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
#         else:
#             for c in xrange(x.shape[1]): # think c is image
#                 startY, startX, flip = self.patch_idx[0], self.patch_idx[1], self.patch_idx[2] # patch coordinates, and whether or not to flip
#                 # print 'startY, startX:', startY, startX
#                 endY, endX = startY + self.inner_size, startX + self.inner_size
#                 maxX, maxY = self.border_size*2, self.border_size*2
#                 patch = y[:, startY:endY, startX:endX, c] # 1st dimension is ':' because take all 3 RGB channels
#                 if flip == 1:
#                     patch = patch[:,:,::-1]
#                 target[:,c] = patch.reshape((self.get_data_dims(),)) # typo?
#             target = augment_illumination(target)
                 
#             if flip == 1:
#                 if self.patch_idx[1] == maxY:
#                     if self.patch_idx[0] == maxX:
#                         self.patch_idx[1] = 0
#                         self.patch_idx[0] = 0
#                         self.patch_idx[2] = 0
#                     else:
#                         self.patch_idx[1] = 0
#                         self.patch_idx[0] += 1
#                         self.patch_idx[2] = 0
#                 else:
#                     self.patch_idx[1] += 1
#                     self.patch_idx[2] = 0
#             else:
#                 self.patch_idx[2] = 1

                
class BasicLeafDataProvider256(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 256

    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        # This converts the data matrix to single precision and makes sure that it is C-ordered
        self.data_dic['data'] = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
        self.data_dic['labels'] = n.require(self.data_dic['labels'].reshape((1,self.data_dic['data'].shape[1])), dtype=n.single, requirements='C')
        return epoch, batchnum, [self.data_dic['data'], self.data_dic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

class BasicLeafDataProvider(LabeledDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 512

    def get_next_batch(self):
        if self.data_dic is None or len(self.batch_range) > 1:
            self.data_dic = self.get_batch(self.curr_batchnum)
        epoch, batchnum = self.curr_epoch, self.curr_batchnum
        self.advance_batch()
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        # This converts the data matrix to single precision and makes sure that it is C-ordered
        self.data_dic['data'] = n.require((self.data_dic['data'] - self.data_mean), dtype=n.single, requirements='C')
        self.data_dic['labels'] = n.require(self.data_dic['labels'].reshape((1,self.data_dic['data'].shape[1])), dtype=n.single, requirements='C')
        return epoch, batchnum, [self.data_dic['data'], self.data_dic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)

class CIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range, init_epoch=1, init_batchnum=None, dp_params={}, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)
        self.data_mean = self.batch_meta['data_mean']
        self.num_colors = 3
        self.img_size = 32
        # Subtract the mean from the data and make sure that both data and
        # labels are in single-precision floating point.
        for d in self.data_dic:
            # This converts the data matrix to single precision and makes sure that it is C-ordered
            d['data'] = n.require((d['data'] - self.data_mean), dtype=n.single, requirements='C')
            d['labels'] = n.require(d['labels'].reshape((1, d['data'].shape[1])), dtype=n.single, requirements='C')

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)
        return epoch, batchnum, [datadic['data'], datadic['labels']]

    # Returns the dimensionality of the two data matrices returned by get_next_batch
    # idx is the index of the matrix. 
    def get_data_dims(self, idx=0):
        return self.img_size**2 * self.num_colors if idx == 0 else 1
    
    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.img_size, self.img_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
class CroppedCIFARDataProvider(LabeledMemoryDataProvider):
    def __init__(self, data_dir, batch_range=None, init_epoch=1, init_batchnum=None, dp_params=None, test=False):
        LabeledMemoryDataProvider.__init__(self, data_dir, batch_range, init_epoch, init_batchnum, dp_params, test)

        self.border_size = dp_params['crop_border']
        self.inner_size = 32 - self.border_size*2
        self.multiview = dp_params['multiview_test'] and test
        self.num_views = 5*2
        self.data_mult = self.num_views if self.multiview else 1
        self.num_colors = 3

        for d in self.data_dic: # for all batches
            d['data'] = n.require(d['data'], requirements='C')
            d['labels'] = n.require(n.tile(d['labels'].reshape((1, d['data'].shape[1])), (1, self.data_mult)), requirements='C')
        
        self.cropped_data = [n.zeros((self.get_data_dims(), self.data_dic[0]['data'].shape[1]*self.data_mult), dtype=n.single) for x in xrange(2)]

        self.batches_generated = 0
        self.data_mean = self.batch_meta['data_mean'].reshape((3,32,32))[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size].reshape((self.get_data_dims(), 1))

    def get_next_batch(self):
        epoch, batchnum, datadic = LabeledMemoryDataProvider.get_next_batch(self)

        cropped = self.cropped_data[self.batches_generated % 2]

        self.__trim_borders(datadic['data'], cropped)
        cropped -= self.data_mean
        self.batches_generated += 1
        return epoch, batchnum, [cropped, datadic['labels']]
        
    def get_data_dims(self, idx=0):
        return self.inner_size**2 * 3 if idx == 0 else 1

    # Takes as input an array returned by get_next_batch
    # Returns a (numCases, imgSize, imgSize, 3) array which can be
    # fed to pylab for plotting.
    # This is used by shownet.py to plot test case predictions.
    def get_plottable_data(self, data):
        return n.require((data + self.data_mean).T.reshape(data.shape[1], 3, self.inner_size, self.inner_size).swapaxes(1,3).swapaxes(1,2) / 255.0, dtype=n.single)
    
    def __trim_borders(self, x, target):
        y = x.reshape(3, 32, 32, x.shape[1])

        if self.test: # don't need to loop over cases
            if self.multiview:
                start_positions = [(0,0),  (0, self.border_size*2),
                                   (self.border_size, self.border_size),
                                  (self.border_size*2, 0), (self.border_size*2, self.border_size*2)]
                end_positions = [(sy+self.inner_size, sx+self.inner_size) for (sy,sx) in start_positions]
                for i in xrange(self.num_views/2):
                    pic = y[:,start_positions[i][0]:end_positions[i][0],start_positions[i][1]:end_positions[i][1],:]
                    target[:,i * x.shape[1]:(i+1)* x.shape[1]] = pic.reshape((self.get_data_dims(),x.shape[1]))
                    target[:,(self.num_views/2 + i) * x.shape[1]:(self.num_views/2 +i+1)* x.shape[1]] = pic[:,:,::-1,:].reshape((self.get_data_dims(),x.shape[1]))
            else:
                pic = y[:,self.border_size:self.border_size+self.inner_size,self.border_size:self.border_size+self.inner_size, :] # just take the center for now
                target[:,:] = pic.reshape((self.get_data_dims(), x.shape[1]))
        else:
            for c in xrange(x.shape[1]): # loop over cases
                startY, startX = nr.randint(0,self.border_size*2 + 1), nr.randint(0,self.border_size*2 + 1)
                endY, endX = startY + self.inner_size, startX + self.inner_size
                pic = y[:,startY:endY,startX:endX, c]
                if nr.randint(2) == 0: # also flip the image with 50% probability
                    pic = pic[:,:,::-1]
                target[:,c] = pic.reshape((self.get_data_dims(),))
    
class DummyConvNetDataProvider(LabeledDummyDataProvider):
    def __init__(self, data_dim):
        LabeledDummyDataProvider.__init__(self, data_dim)
        
    def get_next_batch(self):
        epoch, batchnum, dic = LabeledDummyDataProvider.get_next_batch(self)
        
        dic['data'] = n.require(dic['data'].T, requirements='C')
        dic['labels'] = n.require(dic['labels'].T, requirements='C')
        
        return epoch, batchnum, [dic['data'], dic['labels']]
    
    # Returns the dimensionality of the two data matrices returned by get_next_batch
    def get_data_dims(self, idx=0):
        return self.batch_meta['num_vis'] if idx == 0 else 1
