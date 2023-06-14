1) Make sure the project hoem dir has data folder
2) Data folder contains train, test split and classes outside
3) With each experiment run with pipeline bypass flag in config.INI False , a new folder will be created
4) The new folder contains:
   - npdt folder : all the numpy values are saved
   - XXXasp_indx_dict.json : which contains the aspect and index
   - XXXdata_config json: the configs of data is present
   - XXXdictcord units.json : contains the aspect as key and bounding boxes as values
5) With each experiment run with pipeline bypass flag in config.INI True, the data will be collected from above created folder
