"""Training and evaluation entry points."""
import argparse, os, sys
from pathlib import Path
from model.catalog import ROOT, METHODS, BASELINES, canonical_name

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='stage', required=True)
    sub.add_parser('list')
    prep = sub.add_parser('prepare')
    prep.add_argument('--episodes', type=int, default=3000)
    prep.add_argument('--seeds', nargs='+', type=int, default=[11])
    prep.add_argument('--output', type=Path, default=ROOT / 'evaluation_results/household_multihome_3000')
    for stage in ('train', 'evaluate', 'run'):
        q = sub.add_parser(stage)
        q.add_argument('algorithm', type=canonical_name, choices=METHODS)
        q.add_argument('--seed', type=int)
        q.add_argument('--gpu', type=int)
        q.add_argument('--cpu', action='store_true')
        q.add_argument('--output', type=Path, default=ROOT / 'evaluation_results/household_multihome_3000')
        if stage == 'evaluate':
            q.add_argument('--split', choices=('val', 'test', 'all'), default='all')
    q = sub.add_parser('report')
    q.add_argument('--algorithm', type=canonical_name, choices=METHODS)
    q.add_argument('--output', type=Path, default=ROOT / 'evaluation_results/household_multihome_3000')
    return p

def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.stage == 'list':
        from model.catalog import CATALOG
        for name, item in CATALOG.items():
            print(name + '\t' + item['paper_name'])
        return
    if getattr(args, 'gpu', None) is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    from .protocol import load, prepare, task_protocol
    from .locking import task_lock
    root = args.output.resolve()
    if args.stage == 'prepare':
        with task_lock(root):
            value = prepare(root, args.episodes, args.seeds)
        print('[Ready]', root, '|', value['budget'])
        return
    value = load(root)
    if args.stage == 'report':
        from .report import report
        report(root, value, (args.algorithm,) if args.algorithm else None)
        return
    name = args.algorithm
    if args.cpu and args.gpu is not None:
        p.error('Choose --cpu or --gpu')
    if name in BASELINES:
        if args.seed is not None:
            p.error(name + ' has no training seed')
        if args.stage == 'train':
            p.error(name + ' is an inference-only controller')
        seed = None
        device = None
    else:
        seed = args.seed if args.seed is not None else value['train_seeds'][0] if len(value['train_seeds']) == 1 else None
        if seed not in value['train_seeds']:
            p.error('Supply one registered training seed')
        import torch
        if not args.cpu and (not torch.cuda.is_available()):
            p.error('CUDA unavailable; activate the GPU environment or specify --cpu')
        device = torch.device('cpu' if args.cpu else 'cuda')
        print('[Device]', device, '| visible GPU', os.environ.get('CUDA_VISIBLE_DEVICES', 'all'))
    from .records import run_root
    runroot = run_root(root, seed)
    task = task_protocol(value, seed)
    with task_lock(runroot / name):
        if args.stage in ('train', 'run') and name not in BASELINES:
            from model.train import _train
            _train(runroot, name, task, device)
        if args.stage in ('evaluate', 'run'):
            from .evaluate import evaluate
            splits = ('val', 'test') if getattr(args, 'split', 'all') == 'all' else (args.split,)
            for split in splits:
                evaluate(runroot, name, task, split, device)

def algorithm_main(name, argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ('train', 'evaluate', 'run'):
        return main([args[0], name, *args[1:]])
    if args and args[0] == 'report':
        return main(['report', '--algorithm', name, *args[1:]])
    return main(['evaluate', name, *args])
if __name__ == '__main__':
    main()
