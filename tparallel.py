from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    data = {'a':1,'b':2,'c':3}
    comm.bcast(data, root=0)
else:
    data = None
    struff = comm.bcast(data, root=0)
    print 'rank',rank,struff



# #hello.py
# from mpi4py import MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print "hello world from process ", rank
# tag = {'READY': 1, 'DONE': 2, 'EXIT': 3, 'START': 4}
# print tags['READY']

# mydict = {'a': 1, 'b': 2, 'c': 3}
# tag = {'READY': 1, 'DONE': 2, 'EXIT': 3, 'START': 4}
# items = mydict.items()
# item = items[2]
# print item
# # # test=['apple', 2]
# # # if 1 == tag['READY']:
# # # 	print 'YES'
# # # else:
# # # 	print 'NO'
# # # # for key,value in items:
# # # #     if key == 'b':  # some condition
# # # #         try:
# # # #             key,value = next(items)
# # # #         except StopIteration:
# # # #             break
# # # #     print(key, value)

# # from mpi4py import MPI

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# if rank == 0:
#     data = {'a': 7, 'b': 3.14}
#     comm.send(data, dest=1, tag=11)
# elif rank == 1:
#     data = comm.recv(source=0, tag=11)
#     print data

# def texts_master(comm, size, status, tag):

#   texts = {}
#   texts_norm = {}

#   task_index = 0
#   workers_done = 0
#   workers = size - 1

#   #Gather texts
#   texts_dir = get_text_dir('/tests/test3')
#   #Collect all texts
#   texts = collect_texts(texts_dir)

#   text_list = texts.items()

#   while workers_done < workers:

#     #Master recibe el estado actual de los workers
#     message = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
#     task_tag = status.Get_tag()
#     source = status.Get_source()

#     if task_tag == tag['READY']:
#       if task_index < len(text_list):
#         comm.send(text_list[task_index], dest=source, tag=tag['START'])
#         task_index+=1
#         print 'Collecting document', task_index,'...'
#       else:
#         comm.send(None, dest=source, tag=tag['EXIT'])
#     elif task_tag == tag['DONE']:
#       texts_norm[message[0]] = message[1]    
#     elif task_tag == tag['EXIT']:
#       workers_done += 1
#   return texts_norm


# def texts_worker(comm, rank, status, tag):
#   #Create list of stop words
#   stop_words = init_stop_words('english')
#   comm.send(None, dest=0, tag=tag['READY'])
#   task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
#   task_tag = status.Get_tag()
#   while task_tag!=tag['EXIT']:
#     if task_tag == tag['START']:
#       #Clean and optimize texts for functionality
#       message = normalize_texts(task[0], task[1], stop_words)
#       comm.send(message, dest=0, tag=tag['DONE'])

#     comm.send(None, dest=0, tag=tag['READY'])
#     task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
#     task_tag = status.Get_tag()

#   comm.send(None, dest=0, tag=tag['EXIT'])  


# def cos_distance(vector1, vector2):
#   magnitude_v1 = np.linalg.norm(vector1)
#   magnitude_v2 = np.linalg.norm(vector2)
#   dot_product = np.dot(vector1, vector2)
#   cross_product = magnitude_v1 * magnitude_v2
#   if cross_product != 0:
#     return 1 - (float(dot_product)/cross_product)
#   else:
#     return 0