import numpy as np
import pandas as pd

freq = 500  # частота дискретизации
tolerance = (150 / 1000) * freq  # допустимое временное окно 150 мс


def change_mask(sample):
    """
    приводим маску к виду, где хранится только начало и конец интервала
    :param sample:
    :return:
    """
    p = [[], []]
    qrs = [[], []]
    t = [[], []]
    for i in range(sample.shape[0] - 1):
        if sample[i] != 0 and sample[i + 1] == 0:
            # start_P
            p[0].append(i)
        elif sample[i] != 1 and sample[i + 1] == 1:
            # start_QRS
            qrs[0].append(i)
        elif sample[i] != 2 and sample[i + 1] == 2:
            # start_T
            t[0].append(i)

        if sample[i] == 0 and sample[i + 1] != 0:
            # end_P
            p[1].append(i)
        elif sample[i] == 1 and sample[i + 1] != 1:
            # end_QRS
            qrs[1].append(i)
        elif sample[i] == 2 and sample[i + 1] != 2:
            # end_T
            t[1].append(i)
    return p, qrs, t


def comprassion(mask1, mask2, start_or_end):
    """
    сравнивает два отведения по одному полю
    :param mask1:
    :param mask2:
    :param start_or_end: 0 -- начало интервала, 1 -- конец
    :return:
    """
    tp = []
    fp = []
    fn = []
    error = []

    for p_1 in mask1[start_or_end]:
        flag = False
        for p_2 in mask2[start_or_end]:
            if p_1 + tolerance >= p_2 > p_1 - tolerance:
                tp.append(p_1)
                error.append((p_1 - p_2) / freq)
                mask2[start_or_end].remove(p_2)
                flag = True
                break
        if not flag:
            fp.append(p_1)

    for p_2 in mask2[start_or_end]:
        fn.append(p_2)

    return tp, fp, fn, error


def statistics(y_true, y_pred):
    df_res = pd.DataFrame(
        {'start_p': [0.0, 0.0, 0.0, 0.0, 0.0], 'end_p': [0.0, 0.0, 0.0, 0.0, 0.0], 'start_qrs': [0.0, 0.0, 0.0, 0.0, 0.0],
         'end_qrs': [0.0, 0.0, 0.0, 0.0, 0.0], 'start_t': [0.0, 0.0, 0.0, 0.0, 0.0], 'end_t': [0.0, 0.0, 0.0, 0.0, 0.0]},
        index=['Se', 'PPV', 'F1', 'm', 'sigma^2'])
    df_stat = pd.DataFrame(
        {'start_p': [0, 0, 0], 'end_p': [0, 0, 0], 'start_qrs': [0, 0, 0], 'end_qrs': [0, 0, 0], 'start_t': [0, 0, 0],
         'end_t': [0, 0, 0]}, index=['tp', 'fp', 'fn'])
    df_errors = pd.DataFrame(
        {'start_p': [[]], 'end_p': [[]], 'start_qrs': [[]], 'end_qrs': [[]], 'start_t': [[]], 'end_t': [[]]})
    for j in range(y_pred.shape[0]):
        sample_true = np.argmax(y_true[j], -1)
        sample_pred = preproc(np.argmax(y_pred[j], -1))

        p1, qrs1, t1 = change_mask(sample_pred)
        p2, qrs2, t2 = change_mask(sample_true)

        tp, fp, fn, error = comprassion(p1, p2, 0)
        df_stat.at['tp', 'start_p'] += len(tp)
        df_stat.at['fp', 'start_p'] += len(fp)
        df_stat.at['fn', 'start_p'] += len(fn)
        df_errors.at[0, 'start_p'].extend(error)

        tp, fp, fn, error = comprassion(p1, p2, 1)
        df_stat.at['tp', 'end_p'] += len(tp)
        df_stat.at['fp', 'end_p'] += len(fp)
        df_stat.at['fn', 'end_p'] += len(fn)
        df_errors.at[0, 'end_p'].extend(error)

        tp, fp, fn, error = comprassion(qrs1, qrs2, 0)
        df_stat.at['tp', 'start_qrs'] += len(tp)
        df_stat.at['fp', 'start_qrs'] += len(fp)
        df_stat.at['fn', 'start_qrs'] += len(fn)
        df_errors.at[0, 'start_qrs'].extend(error)

        tp, fp, fn, error = comprassion(qrs1, qrs2, 1)
        df_stat.at['tp', 'end_qrs'] += len(tp)
        df_stat.at['fp', 'end_qrs'] += len(fp)
        df_stat.at['fn', 'end_qrs'] += len(fn)
        df_errors.at[0, 'end_qrs'].extend(error)

        tp, fp, fn, error = comprassion(t1, t2, 0)
        df_stat.at['tp', 'start_t'] += len(tp)
        df_stat.at['fp', 'start_t'] += len(fp)
        df_stat.at['fn', 'start_t'] += len(fn)
        df_errors.at[0, 'start_t'].extend(error)

        tp, fp, fn, error = comprassion(t1, t2, 1)
        df_stat.at['tp', 'end_t'] += len(tp)
        df_stat.at['fp', 'end_t'] += len(fp)
        df_stat.at['fn', 'end_t'] += len(fn)
        df_errors.at[0, 'end_t'].extend(error)


    for index in df_res.columns:
        for i in range(len(df_errors.at[0, index])):
            df_errors.at[0, index][i] = df_errors.loc[0, index][i]*1000
        #df_errors.at[0, index] = df_errors.loc[0, index]/500
        try:
            df_res.at['Se', index] = df_stat.loc['tp', index] / (df_stat.loc['tp', index] + df_stat.loc['fn', index])
        except:
            df_res.at['Se', index] = 0
        try:
            df_res.at['PPV', index] = df_stat.loc['tp', index] / (df_stat.loc['tp', index] + df_stat.loc['fp', index])
        except:
            df_res.at['PPV', index] = 0
        df_res.at['F1', index] =  2 * df_res.loc['Se', index] * df_res.loc['PPV', index] / (df_res.loc['Se', index] + df_res.loc['PPV', index])          
        df_res.at['sigma^2', index] = np.var(df_errors.loc[0, index])
        df_res.at['m', index] = np.mean(df_errors.loc[0, index])
    return df_res

def preproc (sample):
    """
    примитивная подготовка разметки сети: убирает небольшие разрывы интервалов
    :param sample:
    :return:
    """
    for i in range(sample.shape[0]-10):
        if sample[i] !=0:
            if sample[i] != sample[i+1]:
                for j in range(10):
                    if sample[i+j+1] == sample[i]:
                        sample[i+1:i+j] = sample[i]
    return  sample