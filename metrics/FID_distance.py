from ignite.metrics import FID



def compute_fid_score(imgs):
    metric = FID()
    # metric.attach(default_evaluator, "fid")