from .common import *


def stat_task_end(tasks, buff, prop=0.98):
    """
    Get the model checkpoint when prop(ortion) of training examples have been visited
    :param tasks:
    :param buff:
    :return: array of length "tasks" indicating the
    """
    record_cnt = np.zeros(len(tasks))
    total_cnt = np.zeros(len(tasks))
    task_end_index = np.zeros(len(tasks), dtype=np.int32)
    for i, item in enumerate(buff):
        task = item['task'] if 'task' in item else item['task_id']
        if task in tasks:
            task_id = tasks.index(task)
            total_cnt[task_id] += 1
    for i, item in enumerate(buff):
        task = item['task'] if 'task' in item else item['task_id']
        if task in tasks:
            task_id = tasks.index(task)
            record_cnt[task_id] += 1
            if record_cnt[task_id] >= prop * total_cnt[task_id] and task_end_index[task_id] == 0:
                task_end_index[task_id] = i
    return task_end_index


def stat_forgetting_single(task_end_index, tasks, output_dir='runs/mscoco-vlbert-naive-lr0.0001_1', dataset='coco'):
    ckpt_performance = np.zeros(len(tasks))
    fin_performance = None
    iters = [2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, -1] if dataset == 'coco' else \
        [2000, 4000, 6000, 8000, 10000, 12000, 14000, -1]
    task_freq = np.zeros(len(tasks))
    for itr in iters:
        cand_tasks = []
        for tid in range(len(task_end_index)):
            lt = task_end_index[tid]
            # task ends should be after the time step "itr"
            if abs(lt - itr) <= min([abs(lt - x) for x in iters]):
                cand_tasks.append(tid)
        perf = np.zeros((len(tasks),))
        cnts = np.zeros((len(tasks),), dtype=np.int32)
        if itr != -1:
            filename = os.path.join(output_dir, 'results_verbose_model_0_{}.json'.format(itr))
        else:
            filename = os.path.join(output_dir, 'results_verbose_model_00.json')
        data = json.load(open(filename))

        #if dataset == 'flickr':
        #    append_task_id_to_flickr_results(data)

        # compute performance of each task at this time step
        for item in data['records']:
            ppl_list = item['ppl_list']
            mean = np.mean(ppl_list)
            task = item['task']
            tid = tasks.index(task)
            perf[tid] += mean
            cnts[tid] += 1
            task_freq[tid] += 1

        mean_perf = perf / (cnts + 1e-10)

        for tid in cand_tasks:
            ckpt_performance[tid] = mean_perf[tid]

        # performances.append(mean_perf)
        if itr == -1:
            fin_performance = mean_perf
    inc = fin_performance - ckpt_performance

    # do weighted average
    s = (task_freq * inc).sum() / task_freq.sum()

    return s  # inc.mean()


def stat_forgetting_all(methods, dataset_name, task_end_index, tasks, seeds=(0,1,2)):
    res_arr = np.zeros((len(methods), len(seeds)))
    for m, method in enumerate(methods):
        for s, seed in enumerate(seeds):
            try:
                res_arr[m, s] = stat_forgetting_single(task_end_index, tasks, method.format(seed), dataset_name)
            except FileNotFoundError:
                logger.warning(method, seed, 'not found')
        logger.info('Finished evaluating ({}) {}'.format(m, method))
    return res_arr
