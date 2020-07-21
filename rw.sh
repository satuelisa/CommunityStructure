python3 rw.py quiet > tmp
rc=$?
if [[ $rc != 0 ]]; then
    echo "Generation aborted"
    exit
else
    grep -v "#" tmp | awk '{ print NF, $0 }' | sort -n -s -r | cut -d" " -f2- > rw.csv
    rm tmp
    Rscript rw.R gif
    echo "Execution complete"
    open rw.png
fi
