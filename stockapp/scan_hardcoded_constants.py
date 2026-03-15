import os
import ast
import streamlit as st
import pandas as pd


def scan_constants(root_dir="."):

    records = []

    for root, _, files in os.walk(root_dir):

        for file in files:

            if file.endswith(".py"):

                filepath = os.path.join(root, file)

                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        tree = ast.parse(f.read())

                    for node in ast.walk(tree):

                        # Detect CONSTANT assignments
                        if isinstance(node, ast.Assign):

                            for target in node.targets:

                                if isinstance(target, ast.Name):

                                    name = target.id

                                    if name.isupper():

                                        records.append(
                                            {
                                                "file": filepath,
                                                "line": node.lineno,
                                                "constant": name,
                                            }
                                        )

                        # Detect raw string literals
                        if isinstance(node, ast.Constant):

                            if isinstance(node.value, str):

                                if len(node.value) > 5:

                                    records.append(
                                        {
                                            "file": filepath,
                                            "line": node.lineno,
                                            "constant": node.value,
                                        }
                                    )

                except Exception:
                    pass

    return pd.DataFrame(records)


def show_audit():

    st.title("Hard-Coded Data Audit")

    st.write(
        """
        This page scans every `.py` file in the repository and identifies
        possible hard-coded constants, strings, and labels that may violate
        the **Single Source of Truth** principle.
        """
    )

    if st.button("Run Audit"):

        df = scan_constants(".")

        st.success(f"Scan complete — {len(df)} findings")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "Download results as CSV",
            df.to_csv(index=False),
            file_name="hardcode_audit.csv",
        )