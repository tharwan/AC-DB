class Recorder(object):
    
    suffixes = ['all']
    
    def _splitNameSuffix(name):
        if '_' in name:
            last_ = -name[::-1].find('_')
            first = name[:last_-1]
            last = name[last_:]
            return first,last
        else:
            return name,""

    def __init__(self):
        self.recorded = {}

    def addVariable(self,name,value=None):
        name_,suffix = Recorder._splitNameSuffix(name)
        if name in self.recorded:
            raise AttributeError('{} already in Recorder'.format(name))
        elif suffix in Recorder.suffixes:
            raise AttributeError(
                '\"{}\" conflicts with the reserved suffixes: {}'.format(
                    name,Recorder.suffixes))
        else:
            if value is None:
                self.recorded[name] = []
            else:
                self.recorded[name] = [value]
        
            

    def addVariables(self,names,values=None):
        # values can be either a single value for all or a list or None
        

        if not isinstance(values,list):
            for name in names:
                self.addVariable(name,values)
        elif len(values) == len(names):
            for name,value in zip(names,values):
                self.addVariable(name,value)
        elif values is None:
            for name,value in names:
                self.addVariable(name)
        else:
            raise AttributeError('list of values not compatible')

    def historyOf(self,name):
        if name in self.recorded:
            return self.recorded[name]
        else:
            raise AttributeError('{} not in Recorder'.format(name))

    def resetHistoryOf(self,name,value=None):
        
        if value is None:
            value = []
        elif not isinstance(value,list):
            value = [value]

        if name in self.recorded:
            self.recorded[name] = value
        else:
            raise AttributeError('{} not in Recorder'.format(name))


    def __getattribute__(self, name):
        
        if name == 'recorded':
            return super().__getattribute__(name)
        
        else:
            name_,suffix = Recorder._splitNameSuffix(name)
            if name_ in self.recorded and name not in self.recorded:
                if len(self.recorded[name_]) > 0 and not suffix:
                    return self.recorded[name_][-1]
                elif suffix == "all":
                    return self.recorded[name_]
                else:
                    return None
            elif name in self.recorded:
                if len(self.recorded[name]) > 0:
                    return self.recorded[name][-1]
                else:
                    return None
            else:
                return super().__getattribute__(name)

    def __setattr__(self, name, value):
        if name == 'recorded':
            super().__setattr__(name,value)
        else:
            if name in self.recorded:
                self.recorded[name].append(value)
            else:
                super().__setattr__(name,value)
                # raise AttributeError('{} not found in Recorder'.format(name))

