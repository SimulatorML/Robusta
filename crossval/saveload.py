import joblib
import os
import regex

from robusta import utils


__all__ = [
    'save_result',
    'load_result',
    'find_result',
    'list_results',
]




def save_result(result, idx, name, make_submit=True, force_rewrite=False,
                result_path='./results/', submit_path='./submits/',
                silent_mode=None):

    file_name = find_result(idx, result_path)

    if file_name:
        if force_rewrite:
            print('Deleting:\n' + file_name)
            os.remove(file_name)
            print()
        else:
            raise IOError("Model {} already exists!".format(idx))

    print('Creating:')
    path = os.path.join(result_path, '[{}] {}.pkl'.format(idx, name))
    _ = joblib.dump(result, path)
    print(path)

    if make_submit and ('new_pred' in result):

        path = os.path.join(submit_path, '[{}] {}.csv'.format(idx, name))
        pred = result['new_pred']
        pred.to_csv(path, header=True)
        print(path)

        if silent_mode:
            path = os.path.join(submit_path, '[{}S] {}.csv'.format(idx, name))
            pred = POST_PROC[silent_mode](pred)
            pred.to_csv(path, header=True)
            print(path)
    print()



def load_result(idx, result_path='./results/'):

    for fname in os.listdir(result_path):
        if fname.startswith('[{}]'.format(idx)) and fname.endswith('.pkl'):
            path = os.path.join(result_path, fname)
            result = joblib.load(path)

            model_name = fname.split(' ', 1)[1][:-4]
            result.update(idx=idx, model_name=model_name)
            return result

    return None



def find_result(idx, result_path='./results/'):

    for fname in os.listdir(result_path):
        if fname.startswith('[{}]'.format(idx)) and fname.endswith('.pkl'):
            path = os.path.join(result_path, fname)
            return path



def list_results(result_path='./results/'):

    fnames = []

    for fname in os.listdir(result_path):
        if fname.endswith('.pkl'):
            fnames.append(fname)

    return fnames



POST_PROC = {
    'roc_auc': lambda y: 1-y,
}
