from Solver_IMG import *

class ModelPred(Model):
    def __init__(self, **args):
        super(ModelPred, self).__init__(**args)

    def predict_step(self, batch, batch_idx):
        ximg, xoct, y = batch
        ximgs = [ximg, ximg.flip(-1), ximg.flip(-2), ximg.flip(-1, -2),
                ximg.transpose(-1, -2), ximg.transpose(-1, -2).flip(-1), ximg.transpose(-1, -2).flip(-2), ximg.transpose(-1, -2).flip(-1, -2)]
        yhat = 0
        for ximg in ximgs:
            yhat += self(ximg, xoct) / len(ximgs)
        return yhat

    def predict_dataloader(self):
        img_files = sorted(glob.glob("./data/test/images/*/*_crop.jpg"))
        oct_files = sorted(glob.glob("./data/test/images/*/*/*_crop.png"))

        df_img = pd.DataFrame({"img_file": img_files})
        df_img["uid"] = df_img.img_file.apply(lambda x: int(os.path.basename(os.path.dirname(x))))
        df_oct = pd.DataFrame({"oct_file": oct_files})
        df_oct["uid"] = df_oct.oct_file.apply(lambda x: int(os.path.basename(os.path.dirname(x))))
        df_oct = df_oct.iloc[::2]

        df = df_img.merge(df_oct, on = "uid", how = "outer")#.merge(labels, on = "uid", how = "outer")
        df["label"] = 0
        df = df.drop_duplicates(["img_file"]).reset_index(drop = True)

        self.ds_test = self.Data(df, self.trans_valid, is_train = False, **self.args)
        return DataLoader(self.ds_test, self.batch_size, num_workers = 4)

trainer = pl.Trainer(
    gpus = len(gpus.split(",")), 
    precision = 16, amp_backend = "native", amp_level = "O1", 
    accelerator = "dp",
    logger = None,
    progress_bar_refresh_rate = 1
)

ckpts = [
    *glob.glob("./logs/b3ns/*/checkpoints/epoch*")
]

preds = []
for ckpt in ckpts:
    model = ModelPred(**args)
    model = model.load_from_checkpoint(ckpt, strict = False)
    pred = trainer.predict(model)
    pred = torch.cat(pred).softmax(1).detach().cpu().numpy()
    preds.append(pred)

preds = np.stack(preds)
np.save(f"./data/features/test_a.npy", preds)

sub = pd.DataFrame(np.eye(3)[preds.mean(0).argmax(1)].astype(int), columns = ["non", "early", "mid_advanced"])
sub["data"] = np.array(model.predict_dataloader().dataset.df.uid)
sub.data = sub.data.apply(lambda x: f"{x:04}")
sub[["data", "non", "early", "mid_advanced"]].to_csv("Classification_Results.csv", index = False)

np.unique(preds.mean(0).argmax(1), return_counts = True)