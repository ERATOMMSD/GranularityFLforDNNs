import csv

import numpy as np
from tabulate import tabulate
from decimal import Decimal
import os


def t_test_table(data, row_names, col_names, name, save_csv=True):
    assert len(data) == len(row_names) and len(data[0]) == len(col_names)

    if save_csv:
        f = open(f"csv_values/t_tests/{name}.csv", "w")
        writer = csv.writer(f, delimiter=";")
        for row in data:
            writer.writerow(row)

    f = open(f"tables/{name}.tex", "w")
    f.write("\\documentclass{article}" + "\n")
    f.write("\\usepackage{array}" + "\n")
    f.write(r"\usepackage[left = 3cm, right = 5cm, top = 2cm]{geometry}")
    f.write("\\begin{document}" + "\n")
    f.write(r"\newcolumntype{C}[1]{>{\centering\arraybackslash} m{#1cm}}" + "\n")
    print("\n" + name)
    print(tabulate(data))

    f.write(f"\\centering \n {name} \n")
    cols = ""
    for n in col_names:
        cols += f" & {n}"
    header = (r"\begin{tabular}{C{2}|" + " C{2}"*len(col_names) + "}\n" +
              cols +
              "\\\\" + "\n" + r"\hline" + "\n")
    f.write(header)

    for i, row_name in enumerate(row_names):
        text_row = str(row_name)
        for j, e in enumerate(data[i]):
            if e is None:
                text_row += " & -"
                continue
            text_row += " & " + f"({'%.2E' % Decimal(e[0])}, {str(e[1])}, {str(e[2])})"
        f.write(text_row + " \\\\ \n")
    f.write("\\end{tabular}\n\n\\vspace{3cm}")

    f.write("\\end{document}")
    f.close()


def save_table_values(tables, headers, dataset):
    for table, name in tables:
        name = name.replace("=", "")
        values = np.zeros((12, 12))
        table = table[1:]
        for i, row in enumerate(table):
            for j, val in enumerate(row[2:]):
                loss, acc = val.split("\n")
                values[i*2][j] = float(loss)
                values[(i*2)+1][j] = float(acc)
        if not os.path.exists("csv_values/loss_acc_tables"):
            os.mkdir("csv_values/loss_acc_tables")
        np.savetxt(f"csv_values/loss_acc_tables/{dataset}_{name}.csv", values, delimiter=',')


def make_tex(tables, headers, dataset):
    save_table_values(tables, headers, dataset)

    f = open(f"tables/tables-{dataset}.tex", "w")
    f.write("\\documentclass{article}" + "\n")
    f.write("\\usepackage{array}" + "\n")
    f.write("\\usepackage{makecell, multirow}" + "\n")
    f.write(r"\usepackage[left = 3cm, right = 5cm, top = 2cm]{geometry}")
    f.write("\\begin{document}" + "\n")
    f.write(r"\newcolumntype{C}[1]{>{\centering\arraybackslash} m{#1cm}}" + "\n")
    for t, name in tables:
        print("\n" + name)
        print(tabulate(headers))
        print(tabulate(t))

        f.write(f"\\centering \n {name}")
        f.write(r"""
        \begin{tabular}{|m{0.2cm}|m{1.3cm}|C{4}|C{4}|C{4}|}
        \hline
        k & Measure & CIFAR\_1 & CIFAR\_2 & CIFAR\_3 \\
        \hline
        \end{tabular}

        """)

        f.write(r"""
        \centering
        \begin{tabular}{|m{0.2cm}|m{1.3cm}|C{0.65}|C{0.65}|C{0.65}|C{0.65}||C{0.65}|C{0.65}|C{0.65}|C{0.65}
        |C{0.65}|C{0.65}|C{0.65}|C{0.65}|}
        \hline
         &  & T & O & D & R & T & O & D & R & T & O & D & R \\
        \hline
        """)
        for row in t[1:]:
            def multirow(vals):
                return r" & \multirowcell{2}{"+vals[0]+"\\\\"+vals[1]+"}"

            text_row = r"\multirowcell{2}{"+row[0][0:2]+"}"
            for e in row[1:]:
                text_row += multirow(e.split("\n"))
            f.write(text_row + " \\\\ \n \\hline \n")
        f.write("\\end{tabular}\n\n\\vspace{3cm}")

    f.write("\\end{document}")
    f.close()
