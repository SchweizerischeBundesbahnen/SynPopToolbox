from typing import List
from typing import Optional
from typing import Union

import altair as alt
import pandas as pd
from altair.utils.schemapi import Undefined
from altair.utils.schemapi import UndefinedType


class AltairScatterPlot:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: UndefinedType = Undefined,
        columns: Optional[List[str]] = None,
    ):
        """
        Initializes the AltairScatterPlot class with necessary parameters.

        Parameters:
            dataframe (pd.DataFrame): The dataframe containing data for the scatter plot.
            x_col (str): The column name for the x-axis values.
            y_col (str): The column name for the y-axis values.
            title (UndefinedType): The title of the chart.
            columns (Optional[List[str]]): The columns to include in the tooltip. Defaults to all columns.
        """
        self.dataframe = dataframe
        self.x_col = x_col
        self.y_col = y_col
        self.title = title
        self.columns = columns if columns else dataframe.columns.tolist()

    def _create_scatter_plot(self) -> alt.Chart:
        """
        Creates a scatter plot with brush and zoom interactions.

        Returns:
            alt.Chart: The Altair scatter plot chart.
        """
        brush = alt.selection_interval()  # Brush for selection
        interaction = alt.selection_interval(bind="scales")  # Zoom interaction

        return (
            alt.Chart(self.dataframe, title=self.title)
            .mark_point()
            .encode(
                x=alt.X(self.x_col, type="quantitative"),
                y=alt.Y(self.y_col, type="quantitative"),
                tooltip=self.columns,
            )
            .add_selection(brush)
            .add_selection(interaction)
        )

    def _create_data_table(self, brush: alt.Selection) -> alt.HConcatChart:
        """
        Creates data tables linked with the scatter plot via a brush.

        Parameters:
            brush (alt.Selection): The brush selection for filtering the data tables.

        Returns:
            alt.HConcatChart: A concatenated chart of data tables.
        """
        ranked_text = (
            alt.Chart(self.dataframe)
            .mark_text()
            .encode(y=alt.Y("row_number:O", axis=None))
            .transform_window(row_number="row_number()")
            .transform_filter(brush)
            .transform_window(rank="rank(row_number)")
            .transform_filter(alt.datum.rank < 20)
        )

        return alt.hconcat(
            *(
                ranked_text.encode(text=f"{c}:N").properties(title=c)
                for c in self.columns
            )
        )

    def _apply_dropdown_filter(
        self,
        chart: alt.Chart,
        dropdown_col: str,
        dropdown_name: str,
        dropdown_all_option: bool,
    ) -> alt.Chart:
        """
        Applies a dropdown filter to the scatter plot.

        Parameters:
            chart (alt.Chart): The base scatter plot chart.
            dropdown_col (str): The column name for dropdown filtering.
            dropdown_name (str): The name label for the dropdown.
            dropdown_all_option (bool): Include an "All" option in the dropdown.

        Returns:
            alt.Chart: The updated chart with the dropdown filter applied.
        """
        options = self.dataframe[dropdown_col].unique().tolist()
        if dropdown_all_option:
            options = ["All"] + options

        dropdown_selection = alt.selection_single(
            fields=[dropdown_col],
            bind=alt.binding_select(options=options, name=dropdown_name),
            init={dropdown_col: options[0]},
        )

        condition = (
            f"({getattr(dropdown_selection, dropdown_col)}[0] == 'All') || "
            f"({getattr(dropdown_selection, dropdown_col)}[0] == datum.{dropdown_col})"
        )

        return chart.add_selection(dropdown_selection).transform_filter(
            condition if dropdown_all_option else dropdown_selection
        )

    def plot(
        self,
        table: bool = True,
        dropdown_col: Optional[str] = None,
        dropdown_name: str = "Filter Canton",
        dropdown_all_option: bool = True,
    ) -> Union[alt.HConcatChart, alt.Chart]:
        """
        Generates the final scatter plot with optional table and dropdown filter.

        Parameters:
            table (bool): Whether to include a data table with the scatter plot.
            dropdown_col (Optional[str]): The column name for dropdown filtering.
            dropdown_name (str): The name label for the dropdown.
            dropdown_all_option (bool): Include an "All" option in the dropdown.

        Returns:
            Union[alt.HConcatChart, alt.Chart]: The final Altair chart, either a simple scatter plot or a combination with data tables.
        """
        scatter_plot = self._create_scatter_plot()

        if table:
            brush = alt.selection_interval()
            data_table = self._create_data_table(brush)
            scatter_plot = scatter_plot.add_selection(brush)
            scatter_plot = alt.hconcat(scatter_plot, data_table).resolve_legend(
                color="independent"
            )

        if dropdown_col:
            scatter_plot = self._apply_dropdown_filter(
                scatter_plot, dropdown_col, dropdown_name, dropdown_all_option
            )

        return scatter_plot
