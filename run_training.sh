conda activate lisa
final_model_name="$1"
echo "final_model_name $final_model_name "
cd Label0
python lisa.py $final_model_name
cd ../Label1
python lisa.py $final_model_name
cd ../Label2
python lisa.py $final_model_name
cd ../Label3
python lisa.py $final_model_name