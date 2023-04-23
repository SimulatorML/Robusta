import io
import sys
import time

from robusta.utils import logmsg, secfmt

############################################################################################
#######################################Logmsg###############################################
############################################################################################

def test_logmsg():
    # redirect stdout to a StringIO buffer
    buffer = io.StringIO()
    sys.stdout = buffer

    # call the logmsg function with a test message
    test_msg = "This is a test message.\nIt has multiple lines.\n"
    logmsg(test_msg)

    # reset stdout to its original value
    sys.stdout = sys.__stdout__

    # check the output in the buffer
    buffer.seek(0)
    output_lines = buffer.readlines()

    # check that the output has the correct format and content
    assert len(output_lines) == test_msg.count('\n') + 1
    for line in output_lines:
        assert line.startswith("[") and line[9] == "]" and line[10] == " "
        assert line[11:].rstrip() in test_msg.split('\n')

    # pause briefly to avoid overloading the output stream
    time.sleep(0.01)

############################################################################################
########################################Secfmt##############################################
############################################################################################

def test_secfmt():
    assert secfmt(60) == "1 min 0 sec"
    assert secfmt(3600) == "1 h 0 min 0 sec"
    assert secfmt(3661) == "1 h 1 min 1 sec"
    assert secfmt(0.5) == "500 ms"
    assert secfmt(0.001) == "1 ms"
    assert secfmt(0) == "0 ms"