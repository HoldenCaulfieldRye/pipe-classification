
def get_test_error(self):
  next_data = self.get_next_batch(train=False)
  test_outputs = []
  number_tested = 0
  print "========================="
  if self.test_many > 0:
      print "Testing %i batches"%(self.test_many)
  elif self.test_one:    
      print "Testing 1 batch"
  else:
      print "Testing all batches"
  while True:
      data = next_data
      self.start_batch(data, train=False)
      load_next = not self.test_one and ((self.test_many < 0 and data[1] < self.test_batch_range[-1]) or (number_tested < self.test_many))
      if load_next: # load next batch
          print 'loading another test batch'
          next_data = self.get_next_batch(train=False)
          number_tested += 1
      test_outputs += [self.finish_batch()]
      if self.test_only or not self.test_one: # Print the individual batch results for safety
          if self.test_many > 0:
              print "%i/%i\t"%(number_tested,self.test_many),
          print "batch %d: %s" % (data[1], str(test_outputs[-1]))
      if not load_next:
          break
      sys.stdout.flush()
  return self.aggregate_test_outputs(test_outputs)



if __name__ == '__main__':
  test
