from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import ipyaggrid
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from st_aggrid import GridOptionsBuilder


class IpyAGGridTableGenerator:  # pylint: disable=too-many-instance-attributes
    """
    A class to generate an interactive HTML grid table for Jupyter notebooks using a pandas DataFrame.

    Attributes:
        dataframe (pd.DataFrame): The DataFrame to be displayed in the grid.
        colnames (Optional[Union[List[str], Dict[str, str]]]): List or dictionary of column names to use in the grid.
        header_height (int): Height of the header in the grid.
        column_toggles (Optional[Union[List[str], Dict[str, List[str]]]]): Column toggle options for visibility.
        col_widths (Optional[Dict[str, int]]): Custom widths for columns.
        hidden_cols (Optional[List[str]]): List of columns to hide.
        width (Optional[Union[int, str]]): Width of the grid in pixels or percentage.
        theme (str): Theme of the grid.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        dataframe: pd.DataFrame,
        colnames: Optional[Union[List[str], Dict[str, str]]] = None,
        header_height: int = 25,
        column_toggles: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        col_widths: Optional[Dict[str, int]] = None,
        hidden_cols: Optional[List[str]] = None,
        width: Optional[Union[int, str]] = "90%",
        theme: str = "ag-theme-fresh",
    ):
        self.dataframe = dataframe.copy()
        self.colnames = colnames
        self.header_height = header_height
        self.column_toggles = column_toggles
        self.col_widths = col_widths or {}
        self.hidden_cols = hidden_cols or []
        self.width = width
        self.theme = theme

    def _prepare_column_definitions(self) -> List[Dict[str, Any]]:
        """
        Prepares the column definitions for the grid based on the initial parameters.

        Returns:
            List[Dict[str, Any]]: A list containing column definitions.
        """
        if self.colnames:
            if isinstance(self.colnames, list):
                self.dataframe.columns = self.colnames
            elif isinstance(self.colnames, dict):
                self.dataframe = self.dataframe.rename(columns=self.colnames)

        column_definitions = []

        for column in self.dataframe.columns:
            col_def = {"field": column, "hide": column in self.hidden_cols}

            if column in self.col_widths:
                col_def.update(
                    {
                        "width": self.col_widths[column],
                        "suppressSizeToFit": True,
                    }
                )

            column_definitions.append(col_def)

        return column_definitions

    def _create_default_col_def(self) -> Dict[str, Any]:
        """
        Creates the default column definition for the grid.

        Returns:
            Dict[str, Any]: The default column definition.
        """
        return {
            "flex": 1,
            "sortable": True,
            "filter": True,
            "resizable": True,
            "headerComponentParams": {
                "template": (
                    '<div class="ag-cell-label-container" role="presentation">'
                    '  <span ref="eMenu" class="ag-header-icon ag-header-cell-menu-button"></span>'
                    '  <div ref="eLabel" class="ag-header-cell-label" role="presentation">'
                    '    <span ref="eSortOrder" class="ag-header-icon ag-sort-order"></span>'
                    '    <span ref="eSortAsc" class="ag-header-icon ag-sort-ascending-icon"></span>'
                    '    <span ref="eSortDesc" class="ag-header-icon ag-sort-descending-icon"></span>'
                    '    <span ref="eSortNone" class="ag-header-icon ag-sort-none-icon"></span>'
                    '    <span ref="eText" class="ag-header-cell-text" '
                    'role="columnheader" style="white-space: normal;"></span>'
                    '    <span ref="eFilter" class="ag-header-icon ag-filter-icon"></span>'
                    "  </div>"
                    "</div>"
                )
            },
        }

    def _prepare_grid_options(self) -> Dict[str, Any]:
        """
        Prepares the grid options for the ipyaggrid.

        Returns:
            Dict[str, Any]: The grid options.
        """
        column_definitions = self._prepare_column_definitions()

        grid_options = {
            "columnDefs": column_definitions,
            "defaultColDef": self._create_default_col_def(),
            "enableRangeSelection": False,
            "rowSelection": "multiple",
            "headerHeight": self.header_height,
            "pagination": len(self.dataframe) > 50,
            "paginationPageSize": 50,
        }

        return grid_options

    def _create_column_toggle_action(self, cols: List[str]) -> str:
        """
        Creates a JavaScript action for toggling column visibility.

        Args:
            cols (List[str]): List of column names to toggle.

        Returns:
            str: JavaScript code for the toggle action.
        """
        return f"""
            var colNames = "{','.join(cols)}".split(',');
            colNames.forEach(c => {{
                var column = gridOptions.columnApi.getColumn(c);
                gridOptions.columnApi.setColumnVisible(c, !column.isVisible());
            }});
            gridOptions.api.sizeColumnsToFit();
        """

    def _create_column_toggles(self) -> Optional[Dict[str, List[Dict[str, str]]]]:
        """
        Creates the column toggle buttons for the grid.

        Returns:
            Optional[Dict[str, List[Dict[str, str]]]]: The column toggle buttons configuration.
        """
        if self.column_toggles is None:
            return None

        if isinstance(self.column_toggles, list):
            self.column_toggles = {c: [c] for c in self.column_toggles}

        buttons = [
            {"name": group, "action": self._create_column_toggle_action(cols)}
            for group, cols in self.column_toggles.items()
        ]

        return {"buttons": buttons}

    def generate_grid(self) -> ipyaggrid.Grid:
        """
        Generates the grid and returns an ipyaggrid.Grid object.

        Returns:
            ipyaggrid.Grid: The generated grid.
        """
        grid_options = self._prepare_grid_options()

        grid_kwargs: Dict[str, Any] = {
            "grid_data": self.dataframe,
            "grid_options": grid_options,
            "quick_filter": True,
            "export_csv": True,
            "export_excel": False,  # paid feature
            "show_toggle_edit": False,
            "export_mode": "auto",
            "index": False,
            "width": self.width,
            "theme": self.theme,
            "height": self.header_height
            + (600 if len(self.dataframe) > 50 else int(len(self.dataframe) * 28.5)),
        }

        column_toggle_buttons = self._create_column_toggles()
        if column_toggle_buttons:
            grid_kwargs["menu"] = column_toggle_buttons

        return ipyaggrid.Grid(**grid_kwargs)


class StreamlitAGGridTable:
    """
    A class to generate an interactive HTML grid table for Streamlit apps using a pandas DataFrame.

    Attributes:
        dataframe (pd.DataFrame): The DataFrame to be displayed in the grid.
        colnames (Optional[Union[List[str], Dict[str, str]]]): List or dictionary of column names to use in the grid.
        column_toggles (Optional[Union[List[str], Dict[str, List[str]]]]): Column toggle options for visibility.
        col_widths (Optional[Dict[str, int]]): Custom widths for columns.
        hidden_cols (Optional[List[str]]): List of columns to hide.
        width (Union[int, str]): Width of the grid in percentage.
        theme (str): Theme of the grid.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        colnames: Optional[Union[List[str], Dict[str, str]]] = None,
        column_toggles: Optional[Union[List[str], Dict[str, List[str]]]] = None,
        col_widths: Optional[Dict[str, int]] = None,
        hidden_cols: Optional[List[str]] = None,
        width: Union[int, str] = "100%",
        theme: str = "streamlit",
    ):
        self.dataframe = dataframe.copy()
        self.colnames = colnames
        self.column_toggles = column_toggles
        self.col_widths = col_widths or {}
        self.hidden_cols = hidden_cols or []
        self.width = width
        self.theme = theme
        self._apply_colnames()

    def _apply_colnames(self):
        """Apply custom column names if provided."""
        if self.colnames:
            if isinstance(self.colnames, list):
                self.dataframe.columns = self.colnames
            elif isinstance(self.colnames, dict):
                self.dataframe.rename(columns=self.colnames, inplace=True)

    def _get_grid_options(self) -> Dict[str, Any]:
        """Build grid options using GridOptionsBuilder."""
        gob = GridOptionsBuilder.from_dataframe(self.dataframe)
        gob.configure_default_column(
            editable=True,
            groupable=True,
            resizable=True,
            sortable=True,
            filterable=True,
        )

        for column, width in self.col_widths.items():
            gob.configure_column(column, width=width)

        for column in self.hidden_cols:
            gob.configure_column(column, hide=True)

        if len(self.dataframe) > 50:
            gob.configure_pagination(
                paginationAutoPageSize=False, paginationPageSize=50
            )

        if self.column_toggles:
            gob.configure_side_bar()

        return gob.build()

    def display_grid(self):
        """Displays the grid using streamlit-aggrid."""
        grid_options = self._get_grid_options()
        AgGrid(
            self.dataframe,
            gridOptions=grid_options,
            height=600 if len(self.dataframe) > 50 else len(self.dataframe) * 30 + 50,
            width=self.width,
            theme=self.theme,
        )

    def download_button(self, label: str):
        """
        Creates a download button for the DataFrame as a CSV file.

        Args:
            label (str): Label for the download button.
        """
        csv_data = self.dataframe.to_csv(index=False).encode()
        st.download_button(
            label=label, data=csv_data, file_name="data.csv", mime="text/csv"
        )


def main():
    # Example DataFrame
    df = pd.DataFrame(
        {
            "A": range(1, 101),
            "B": [f"Value {i}" for i in range(1, 101)],
            "C": [i % 5 for i in range(1, 101)],
        }
    )

    # Filtering example
    search_term = st.text_input("Search:", "")
    if search_term:
        filtered_df = df[
            df.apply(
                lambda row: search_term.lower()
                in row.astype(str).str.lower().to_string(),
                axis=1,
            )
        ]
    else:
        filtered_df = df

    # Create and display grid
    grid_generator = StreamlitAGGridTable(
        dataframe=filtered_df,
        colnames={"A": "Column A", "B": "Column B", "C": "Column C"},
        column_toggles={"Toggle A": ["A"], "Toggle B": ["B"]},
        col_widths={"A": 100, "B": 150},
        hidden_cols=["C"],
    )
    grid_generator.display_grid()

    # Add download button
    grid_generator.download_button("Download CSV")


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    main()
