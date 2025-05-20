import os
import torch
import yaml
import argparse
from core.dataset import MMDataEvaluationLoader
from models.TFMamba import build_model
from core.metric import MetricsTop

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument('--config_file', type=str, default='')
opt = parser.parse_args()
print(opt)


def main():
    config_file = 'configs/eval_mosi.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)
    seed = args['base']['seed']
    dataset_name = args['dataset']['datasetName']


    model = build_model(args).to(device)
    metrics = MetricsTop(train_mode=args['base']['train_mode']).getMetics(dataset_name)

    missing_rate_list = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # missing_rate_list = [1.0]

    all_results = {
        'missing_rates': missing_rate_list,
        'metrics': {}
    }
    if dataset_name == 'sims':
        for key_eval in ['Mult_acc_2', 'Mult_acc_3', 'Mult_acc_5', 'MAE']:
            print(f"\n=== {key_eval} Metrics Across All Missing Rates ===")
            for cur_r in missing_rate_list:
                test_results_list = []
                for _, cur_seed in enumerate([seed]):
                    best_ckpt = os.path.join(f'ckpt/{dataset_name}/best_{key_eval}_{cur_seed}.pth')
                    model.load_state_dict(torch.load(best_ckpt)['state_dict'])
                    args['base']['missing_rate_eval_test'] = cur_r  # Set missing rate

                    dataLoader = MMDataEvaluationLoader(args)

                    test_results_cur_seed = evaluate(model, dataLoader, metrics)
                    test_results_list.append(test_results_cur_seed)

                if key_eval == 'Mult_acc_2':
                    Mult_acc_2_avg = test_results_list[0]['Mult_acc_2']
                    F1_score_avg = test_results_list[0]['F1_score']
                    print(
                        f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {Mult_acc_2_avg}, F1_score_avg: {F1_score_avg}')
                    # 存储结果
                    if 'Mult_acc_2' not in all_results['metrics']:
                        all_results['metrics']['Mult_acc_2'] = []
                    if 'F1_score' not in all_results['metrics']:
                        all_results['metrics']['F1_score'] = []
                    all_results['metrics']['Mult_acc_2'].append(Mult_acc_2_avg)
                    all_results['metrics']['F1_score'].append(F1_score_avg)

                elif key_eval == 'Mult_acc_3':
                    Mult_acc_3_avg = test_results_list[0]['Mult_acc_3']
                    print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_3_avg: {Mult_acc_3_avg}')
                    if 'Mult_acc_3' not in all_results['metrics']:
                        all_results['metrics']['Mult_acc_3'] = []
                    all_results['metrics']['Mult_acc_3'].append(Mult_acc_3_avg)

                elif key_eval == 'Mult_acc_5':
                    Mult_acc_5_avg = test_results_list[0]['Mult_acc_5']
                    print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {Mult_acc_5_avg}')
                    if 'Mult_acc_5' not in all_results['metrics']:
                        all_results['metrics']['Mult_acc_5'] = []
                    all_results['metrics']['Mult_acc_5'].append(Mult_acc_5_avg)

                elif key_eval == 'MAE':
                    MAE_avg = test_results_list[0]['MAE']
                    Corr_avg = test_results_list[0]['Corr']
                    print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {MAE_avg}, Corr_avg: {Corr_avg}')
                    if 'MAE' not in all_results['metrics']:
                        all_results['metrics']['MAE'] = []
                    if 'Corr' not in all_results['metrics']:
                        all_results['metrics']['Corr'] = []
                    all_results['metrics']['MAE'].append(MAE_avg)
                    all_results['metrics']['Corr'].append(Corr_avg)
    else:
        for key_eval in ['Has0_acc_2', 'Non0_acc_2', 'Mult_acc_5', 'Mult_acc_7', 'MAE']:
            print(f"\n=== {key_eval} Metrics Across All Missing Rates ===")
            for cur_r in missing_rate_list:
                test_results_list = []
                for _, cur_seed in enumerate([seed]):
                    best_ckpt = os.path.join(f'ckpt/{dataset_name}-save/best_{key_eval}_{cur_seed}.pth')
                    model.load_state_dict(torch.load(best_ckpt)['state_dict'])
                    args['base']['missing_rate_eval_test'] = cur_r  # Set missing rate

                    dataLoader = MMDataEvaluationLoader(args)

                    test_results_cur_seed = evaluate(model, dataLoader, metrics)
                    test_results_list.append(test_results_cur_seed)

                if key_eval == 'Has0_acc_2':
                    Has0_acc_2_avg = test_results_list[0]['Has0_acc_2']
                    Has0_F1_score_avg = test_results_list[0]['Has0_F1_score']
                    print(
                        f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_2_avg: {Has0_acc_2_avg}, F1_score_avg: {Has0_F1_score_avg}')
                    if 'Has0_acc_2' not in all_results['metrics']:
                        all_results['metrics']['Has0_acc_2'] = []
                    if 'Has0_F1_score' not in all_results['metrics']:
                        all_results['metrics']['Has0_F1_score'] = []
                    all_results['metrics']['Has0_acc_2'].append(Has0_acc_2_avg)
                    all_results['metrics']['Has0_F1_score'].append(Has0_F1_score_avg)

                elif key_eval == 'Non0_acc_2':
                    Non0_acc_2_avg = test_results_list[0]['Non0_acc_2']
                    Non0_F1_score_avg = test_results_list[0]['Non0_F1_score']
                    print(
                        f'key_eval: {key_eval}, missing rate: {cur_r}, Non0_acc_2_avg: {Non0_acc_2_avg}, Non0_F1_score_avg: {Non0_F1_score_avg}')
                    if 'Non0_acc_2' not in all_results['metrics']:
                        all_results['metrics']['Non0_acc_2'] = []
                    if 'Non0_F1_score' not in all_results['metrics']:
                        all_results['metrics']['Non0_F1_score'] = []
                    all_results['metrics']['Non0_acc_2'].append(Non0_acc_2_avg)
                    all_results['metrics']['Non0_F1_score'].append(Non0_F1_score_avg)

                elif key_eval == 'Mult_acc_5':
                    Mult_acc_5_avg = test_results_list[0]['Mult_acc_5']
                    print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_5_avg: {Mult_acc_5_avg}')
                    if 'Mult_acc_5' not in all_results['metrics']:
                        all_results['metrics']['Mult_acc_5'] = []
                    all_results['metrics']['Mult_acc_5'].append(Mult_acc_5_avg)

                elif key_eval == 'Mult_acc_7':
                    Mult_acc_7_avg = test_results_list[0]['Mult_acc_7']
                    print(f'key_eval: {key_eval}, missing rate: {cur_r}, Mult_acc_7_avg: {Mult_acc_7_avg}')
                    if 'Mult_acc_7' not in all_results['metrics']:
                        all_results['metrics']['Mult_acc_7'] = []
                    all_results['metrics']['Mult_acc_7'].append(Mult_acc_7_avg)

                elif key_eval == 'MAE':
                    MAE_avg = test_results_list[0]['MAE']
                    Corr_avg = test_results_list[0]['Corr']
                    print(f'key_eval: {key_eval}, missing rate: {cur_r}, MAE_avg: {MAE_avg}, Corr_avg: {Corr_avg}')
                    if 'MAE' not in all_results['metrics']:
                        all_results['metrics']['MAE'] = []
                    if 'Corr' not in all_results['metrics']:
                        all_results['metrics']['Corr'] = []
                    all_results['metrics']['MAE'].append(MAE_avg)
                    all_results['metrics']['Corr'].append(Corr_avg)


    # 计算并打印平均值
    print("\n=== Average Metrics Across All Missing Rates ===")
    for metric_name, values in all_results['metrics'].items():
        if values:
            avg_value = sum(values) / len(values)
            print(f"Average {metric_name}: {avg_value:.4f}")


def evaluate(model, eval_loader, metrics):
    y_pred, y_true = [], []

    model.eval()
    for cur_iter, data in enumerate(eval_loader):
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))
        sentiment_labels = data['labels']['M'].to(device)

        with torch.no_grad():
            out = model((None, None, None), incomplete_input)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(sentiment_labels.cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)
    return results
if __name__ == '__main__':
    main()