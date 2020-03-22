import psytrance
import trendsdata
import proxies
import argparse
import neuralnet

"""
kw-tmpl is copied to kw-trends
-> do not edit kw-trends
to add new topic/keyword files
create a new topic in kw-tmpl and a file lang.txt with keywords per line;
see below

Modes:
    populate 

       
    proxy-server
        proxy
    
    update-trends
        python3 -m trending update-trends
    
Returns:
    0


"""
MODES = ["topicalize", "proxies", "trends", "neural", "keywordify"]

def main():

    ap = argparse.ArgumentParser("coroner.\n Uses Google Trends and a time series prediction GRU network to predict regional infections based on keyword searches live.\nits pretty proof of concept ish atm")

    ap.add_argument("mode",choices=MODES,\
        help="topicalize: populate a given topic for all languages using an existing keyword file written in just one\
            \nproxies: run proxy collection server so larger bag of keywords can be tracked by 'trends'\
            \ntrends: run trends collection server to sequentially update google trends data\
            \nneural: train or eval neural network\
            \nkeywordify: given a topic string, generate n keywords that should help the neural net")

    args = ap.parse_args(sys.argv[:1])

    if args.mode in ["topicalize", "keywordify", "trends"]:
        raise NotImplementedError()
    else:
        assert args.mode in MODES #LAST SAFETY BEFORE USER BREACH DO NOT REMOVE
        eval(f"{args.mode}()")

def proxies():
    import proxies as p
    PROXY_ACTIONS = ["retrieve","run"]
    ap_ = argparse.ArgumentParser("Running in proxies mode")
    ap_.add_argument("action", choices=PROXY_ACTIONS, help="\
        retrieve: retrieve saved proxy locations\
        \nrun: run server, checking back for new ips every <interval> minutes")
    ap_.add_argument("--interval", type=int,help="minutes to wait between updates")
    args_ = ap_.parse_args(sys.argv[0]+sys.argv[2:])

    print(f"proxies action was {args_.action}")
    raise NotImplementedError("TODO: implement retrieve and run with keywords, figure out how to use package like joey import???")

    
def topicalize():
    """
    topic is the string of a directory name within 
    kw-tmpl containing one keyword file in named by its google translate language
    e.g. en.txt with one keyword per line e.g. finance/en.txt: "crisis\ncrash\nbailout"
    """ 

    raise NotImplementedError
def trends():        
    raise NotImplementedError
def neural():
    raise NotImplementedError
def keywordify():
    raise NotImplementedError




if __name__ == "__main__":
    main()

