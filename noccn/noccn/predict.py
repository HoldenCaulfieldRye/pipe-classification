import csv
from multiprocessing import Process
import sys
import time

import numpy as np
import cPickle as pickle

from ccn import convnet
from ccn import options
from script import get_sections
from script import run_model


def make_predictions(net, data, labels, num_classes):
    data = np.require(data, requirements='C')
    labels = np.require(labels, requirements='C')
    preds = np.zeros((data.shape[1], num_classes), dtype=np.single)
    labels_stub = np.zeros((1, data.shape[1]), dtype=np.single)
    softmax_idx = net.get_layer_idx('probs', check_type='softmax')
    net.libmodel.startFeatureWriter( [data, labels_stub, preds], softmax_idx)
    net.finish_batch()
    if net.multiview_test:
        num_views = net.test_data_provider.num_views
        processed_preds = np.zeros((labels.shape[1],num_classes))
        for image in range(0,labels.shape[1]):
            tmp_preds = preds[image::labels.shape[1]]
            processed_preds[image] = tmp_preds.T.mean(axis=1).reshape(tmp_preds.T.shape[0],-1).T
        preds = processed_preds    
    return preds, labels


class PredictConvNet(convnet.ConvNet):
    _predictions = None
    _option_parser = None
    csv_fieldnames = ''

    def make_predictions(self):
        num_classes = self.test_data_provider.get_num_classes()
        num_batches = len(self.test_data_provider.batch_range)
        classes = self.test_data_provider.batch_meta['label_names']
        self.performance_dict = dict((_class,{'top-1':0,'top-5':0,'top-10':0,'total':0}) for _class in classes)
        for batch_index in range(num_batches):
            t0 = time.time()
            epoch, batchnum, (data, labels) = self.get_next_batch(train=False)
            preds, labels = make_predictions(self, data, labels, num_classes)
            top_results = np.argsort(preds,axis=1).T[::-1].T[:,:10]
            result_row = 0
            for label in labels[0]:
                label = int(label)
                top = list(top_results[result_row])
                try:
                    index = top.index(label)
                except:
                    index = -1
                self.performance_dict[classes[label]]['total']+=1
                if index == 0:
                    self.performance_dict[classes[label]]['top-1']+=1
                if index < 5 and index >= 0:    
                    self.performance_dict[classes[label]]['top-5']+=1
                if index < 10 and index >= 0:    
                    self.performance_dict[classes[label]]['top-10']+=1
                result_row += 1
            print "%i/%i:\tPredicted %s cases in %.2f seconds."%(batch_index+1,num_batches,len(labels[0]),time.time()-t0)
            sys.stdout.flush()
        return self.performance_dict


    def report(self):
        print 'Class Name;Num Correct Top-1;Num Correct Top-5;Num Total',
        print ';Top-1 Error Rate;Top-5 Error Rate'
        total_top1 = 0.0
        total_top5 = 0.0
        total_total = 0.0
        for key in self.performance_dict:
            t1 = self.performance_dict[key]['top-1']
            t5 = self.performance_dict[key]['top-5']
            tot = self.performance_dict[key]['total']
            total_top1+=t1
            total_top5+=t5
            total_total+=tot
            if tot > 0:
                err1 = (tot-t1)/float(tot)
                err5 = (tot-t5)/float(tot)
                print '%s;%i;%i;%i;%.05f;%.05f'%(key,t1,t5,tot,err1,err5)
            else:
                print '%s;%i;%i;%i;N/A;N/A'%(key,t1,t5,tot)
        if total_total > 0:
            err1 = (total_total-total_top1)/float(total_total)
            err5 = (total_total-total_top5)/float(total_total)
            print 'Total;%i;%i;%i;%.05f;%.05f'%(total_top1,total_top5,total_total,err1,err5)
        else:
            print 'Total;%i;%i;%i;N/A;N/A'%(total_top1,total_top5,total_total)


    def start(self):
        self.op.print_values()
        self.make_predictions()
        save_file = open(self.op_write_predictions_file,'wb')
        pickle.dump(self.performance_dict,save_file)
        save_file.close()
        if self.op_report:
            self.report()
        sys.exit(0)

    @classmethod
    def get_options_parser(cls):
        if cls._option_parser is not None:
            return cls._option_parser

        op = convnet.ConvNet.get_options_parser()
        op.add_option("write-preds", "op_write_predictions_file",
                      options.StringOptionParser,
                      "Write predictions to this file")
        op.add_option("report", "op_report",
                      options.BooleanOptionParser,
                      "Do a little reporting?")

        cls._option_parser = op
        return cls._option_parser


def console(net=PredictConvNet):
    cfg = sys.argv.pop(1)
    n_sections = len([s for s in get_sections(cfg) if s.startswith('predict')])
    for section in get_sections(cfg):
        if section.startswith('predict'):
            print "=" * len(section)
            print section
            print "=" * len(section)

            # run in a subprocess because of clean-up
            if n_sections > 1:
                p = Process(
                    target=run_model,
                    args=(net, section, cfg),
                    )
                p.start()
                p.join()
            else:
                run_model(net, section, cfg)
