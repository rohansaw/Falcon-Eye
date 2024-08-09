import os
import json
import numpy as np

prediction_files = {
    #"MHSA-ADD-4-64dim.json": "MHSA-4",
    # "MHSA-ADD-4-32dim.json": "MHSA-4 (32channels)",
    # "MHSA-ADD-4-SW.json": "MHSA-4",
    # "MHSA-ADD-8-SW.json": "MHSA-8",
    # "SIM-AM-SKIP-SW.json": "SIM-AM",
    # "MobileOne-SW.json": "None",
    # "predictions_24.json.json": "new",
    # "predictions_25.json.json": "new2"
    
    # Attention Experiments
    # "Final-MobileOne-SW.json": "None",
    # "Final-MHSA-8-smalllIMg-SW.json": "MHSA-8",
    # "Final-MHSA-4-reducedDim-SW-smallImg.json": "MHSA-4 (32 channels)",
    # "Final-MHSA-4-smallImg-SW.json": "MHSA-4",
    "Final-SIMAM-smallImg.json": "SIM-AM",
    
    #"POG-GSD002-NoCrop.json": "2cm/px",
    #"POG-GSD004-NoCrop.json": "4cm/px",
    #"POG-noGSD-Crop.json": "No resize, Crop Aug",
    #"POG-noGSD-noCrop.json": "No resize, No Crop Aug",
    
    # "POG-GSD008-noCrop.json": "8cm/px",
    # "POG_002-new.json": "2cm/px new",
    # "POG-GSD004-new.json": "4cm/px new",
    # "POG-noGSD-noCrop-new.json": "none, no-crop aug",
    # "POG-noGSD-crop-new.json": "none, crop aug",
    
    # Backbones Experiments
    # "Final-MobileOne-SW.json": "MobileOne (SW)",
    # "MobileNet-SW-fixed.json": "MobileNet (SW)",
    # "MobileNet-SDS.json": "MobileNet (SDS)",
    # "Final-MobileOne-SW-NoRep.json": "MobileOne No-Rep (SW)",
    # "MobileOne-SDS-final-Rep.json": "MobileOne (SDS)",
    # "MobileOne-SDS-final-NoRep.json": "MobileOne No-Rep (SDS)",
    
    # Learning from altitude Experiments
    #"Final-MobileOne-SW.json": "No altitude info",
    #"SW-with-altitude_1.json": "Altitude concat",
    # "SW-with-altitude_add.json": "Altitude add",
    #"SW-with-altitude-modulate.json": "Altitude modulate",
    #"SW-with-altitude-modulate_wrongModel.json": "Altitude modulate wrong",
    
    "Final-SIMAM-smallImg.json": "SIM-AM",
    "predictions_67.json.json": "MobileNet (SW) + SIMAM + AltCat",
    
}

# prediction_files = {}

# for f in os.listdir("predictions_store_baselines"):
#     if "sds" in f:
#         prediction_files[f] = f.split(".")[0]
        
# print(prediction_files)


def compute_ap_by_size(precisions, recalls_by_size):
    precisions = np.array(precisions)
    for size in recalls_by_size:
        recalls = np.array(recalls_by_size[size])
        AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        print(f"AP {size}: ", AP)
        
        # print best f1 for this size   
        # f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        # best_f1 = np.max(f1_scores)
        # print(f"Best F1 {size}: ", best_f1)

def compute_ar_by_size(recalls_by_size):
    for size in recalls_by_size:
        recalls = np.array(recalls_by_size[size])
        AR = np.mean(recalls)
        print(f"AR {size}: {AR}")


results = []
names = []


for fn in os.listdir("predictions_store"):
    if  fn not in prediction_files:
        continue
    with open(f"predictions_store/{fn}", "r") as f:
        # if not "POG" in fn:
        #     continue
        
        # if not "sliced" in fn:
        #     continue
        
        print("------------")
        print("")
        res = json.load(f)
        
        while res["recalls"][-1] == 0 and res["precisions"][-1] == 0:
            res["recalls"] = res["recalls"][:-1]
            res["precisions"] = res["precisions"][:-1]
            if "recalls_by_size" in res:
                for rec in res["recalls_by_size"]:
                    res["recalls_by_size"][rec] = res["recalls_by_size"][rec][:-1]
        
        if not 0 in res["recalls"]:
            print("appending")
            res["recalls"].append(0)
            res["precisions"].append(1)
            if "recalls_by_size" in res:
                for rec in res["recalls_by_size"]:
                    res["recalls_by_size"][rec].append(0)
            
        
        results.append(res)
        # conf_idx = 50
        # p = res["precisions"][conf_idx]
        # r = res["recalls"][conf_idx]
        # f1 = 2 * (p * r) / (p + r)
     
        print(res["model_name"])
        #print(prediction_files[fn])
        # print("precision: ", p) 
        # print("recall: ", r)
        # print("f1: ", f1)
        precisions = np.array(res["precisions"])
        recalls = np.array(res["recalls"])
        AP = np.sum((recalls[:-1] - recalls[1:]) * precisions[:-1])
        print("AP: ", AP)
        
        # get best f1 from precision recall curve
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        best_f1 = np.max(f1_scores)
        print("Best F1: ", best_f1)
        
        # print precision and recall for max f1
        max_f1_idx = np.argmax(f1_scores)
        print("Precision: ", precisions[max_f1_idx])
        print("Recall: ", recalls[max_f1_idx])
        
        # print precision where recall first hits 0.9
        if max (recalls) >= 0.9:
            recall_idx = np.where(recalls >= 0.9)[0][-1]
            
            print("Precision at 0.9 recall: ", precisions[recall_idx])
        
        if "recalls_by_size" in res:
            compute_ap_by_size(precisions, res["recalls_by_size"])
            compute_ar_by_size(res["recalls_by_size"])
        
        # print(len(json.load(f)["precisions"]))
        # print(len(json.load(f)["recalls"]))
        names.append(fn.split(".")[0])
   
   
import matplotlib.pyplot as plt
for size in ["tiny", "small", "medium", "large"]:
    plt.figure(figsize=(8, 6)) 
    for res, name in zip(results, names):
        precisions = np.array(res["precisions"])
        recalls = np.array(res["recalls_by_size"][size])
        color = "blue" if "mobilenet" in name.lower() else "red"
        linestyle = "--" if  "sds" in name.lower()else "-"
        plt.plot(res["recalls"], res["precisions"], label=name, linestyle=linestyle)
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {size} Objects Only')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"precision_recall_curve_{size}.png")
    exit()


plt.figure(figsize=(8, 6))     
for res, name in zip(results, names):
    # if "predictions" in name:
    #     continue
    color = "blue" if "mobilenet" in name.lower() else "red"
    linestyle = "--" if  "sds" in name.lower()else "-"
    plt.plot(res["recalls"], res["precisions"], label=name, linestyle=linestyle)
    #plt.plot(res["recalls"], res["precisions"], label=prediction_files[name+".json"], linestyle=linestyle)
    
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.savefig("precision_recall_curve.png")
