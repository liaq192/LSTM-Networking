#This is LSTM application on CPU usage Dataset#

Dataset URL :http://gwa.ewi.tudelft.nl/datasets/gwa-t-1-das2
Fetched Date: 24th Dec 2019

Complete dataset contains following fields

JobID,SubmitTime,WaitTime,RunTime,NProc,UsedCPUTime,UsedMemory,ReqNProcs,ReqTime,ReqMemory,Status,UserID,GroupID,ExecutableID,QueueID,PartitionID,OrigSiteID,LastRunSiteID,JobStructure,JobStructureParams,UsedNetwork,UsedLocalDiskSpace,UsedResources,ReqPlatform,ReqNetwork,ReqLocalDiskSpace,ReqResources,VOID,ProjectID

Only using following fields with first commit.

JobID,SubmitTime,WaitTime,RunTime,NProc,UsedCPUTime,UsedMemory, UserID,


UserID: Category therefore is converted into OneHotEncoding before usage. 


