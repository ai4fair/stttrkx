#!/bin/bash

# Script to find and move certain files, doing logistic is efficient way.

# Move clean & complex events from trkx_from_gnn to trkx_clean and trkx_compl
# directories. This is intended for trkx_reco_eval.sh on these directories.

input_dir="run/trkx_from_gnn"
clean_dir="run/trkx_from_gnn/clean"
compl_dir="run/trkx_from_gnn/compl"

rm -rf $clean_dir
rm -rf $compl_dir

mkdir -p $clean_dir
mkdir -p $compl_dir

# File Lists
clean_l=(5014 5016 5017 5023 5038 5039 5043 5044 5050 5051 5053 5055 5065 5078 5080 5082 5083 5086 5090 5091 5111 5115 5118 5124 5127 5133 5152 5159 5160 5163 5165 5169 5180 5181 5196 5202 5205 5216 5223 5226 5227 5233 5249 5264 5266 5271 5273 5276 5296 5300 5510 5511 5513 5560 5563 5565 5566 5575 5576 5578 5584 5587 5597 5601 5609 5615 5637)

compl_l=(5005 5006 5007 5008 5009 5015 5018 5021 5024 5025 5027 5028 5029 5030 5031 5032 5034 5037 5041 5045 5048 5049 5052 5054 5056 5060 5061 5062 5063 5064 5084 5092 5094 5099 5100 5106 5107 5108 5112 5119 5126 5128 5130 5131 5136 5138 5150 5151 5153 5154 5157 5166 5170 5171 5173 5175 5176 5182 5184 5185 5191 5197 5204 5206 5208 5210 5217 5222 5248 5255 5265 5274 5280 5500 5501 5504 5505 5506 5508 5509 5514 5550 5611 5619 5621 5622 5623)

# Move Listed Files
for f in ${clean_l[@]}; do
    echo "Copying Clean File: $f"
    cp $input_dir/$f $clean_dir

done

# Move Listed Files
for f in ${compl_l[@]}; do
    echo "Copying Complex File: $f"
    cp $input_dir/$f $compl_dir

done
