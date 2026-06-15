"""Improved Tree diagram class with flexible spacing.

Based on original TreeDiagram by Adam Michael Bauer
Enhanced to allow for more spacing by TM
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Rectangle

from ..config import DEFAULT_BASE_YEAR, DEFAULT_DECISION_TIMES

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class TreeDiagramSpaced:
    """Enhanced tree diagram class with flexible spacing.

    Makes plot of CAP6 model output in path dependent binomial tree
    form with customizable spacing and progressive font sizing for better
    readability of price data.

    Parameters
    ----------
    N_periods: int
        number of periods to display
    data: nd array
        the data we want plotted
    path_notation: bool
        is the data in path notation? i.e., does the data have shape (32,6)?
    plot_title: string
        title of plot (optional - use empty string '' for no title)
    figsize: tuple
        size of figure
    save_fig: bool
        dictates whether we should save the figure or not
    fig_name: string
        filename if save_fig is turned on (without extension)
    is_cost: bool
        if True, puts a dollar sign in front of the text in boxes
    decision_times: list
        list of decision times (years from base year) for x-axis labels
    box_scale_mode: string
        'progressive' - boxes get smaller in later periods
        'uniform' - all boxes same size
        'binary' - early periods large, last period small
    font_scale_mode: string
        'progressive' - font size decreases smoothly across periods
        'binary' - early periods large font, last period small font
        'uniform' - all same font size
    spacing_multiplier: float
        multiplier for overall vertical spacing (increase to spread out tree)
    base_year: int
        base year for x-axis labels
    even_spacing_last_period: bool
        if True, spreads boxes in the last period evenly across vertical space
        instead of using the grouped structure (default False)
    """

    def __init__(self, N_periods, data, path_notation, plot_title=None,
                 figsize=(14, 10), save_fig=True, fig_name='tree_plot_spaced',
                 is_cost=False, decision_times=None, box_scale_mode='progressive',
                 font_scale_mode='progressive', spacing_multiplier=1.5,
                 base_year=DEFAULT_BASE_YEAR, even_spacing_last_period=False):
        
        self.N_periods = N_periods
        self.N_nodes = int(2**self.N_periods - 1)
        self.path_notation = path_notation
        self.is_cost = is_cost
        self.box_scale_mode = box_scale_mode
        self.font_scale_mode = font_scale_mode
        self.spacing_multiplier = spacing_multiplier
        self.base_year = base_year
        self.even_spacing_last_period = even_spacing_last_period
        
        # Store decision times for custom year labels
        if decision_times is None:
            self.decision_times = DEFAULT_DECISION_TIMES
        else:
            self.decision_times = decision_times

        # Convert path notation to node notation if needed
        if self.path_notation:
            self.data = self.get_path_to_node(data)
        else:
            self.data = data

        # Initialize coordinate arrays
        self.node_coords = np.zeros((self.N_nodes, 2))
        self.x_lines = np.zeros((self.N_nodes-1, 2))
        self.y_lines = np.zeros((self.N_nodes-1, 2))
        self.track_additions = np.array([
            [1,2,1,2], 
            [1,2,3,1,2,3,4,5,6,4,5,6],
            [1,2,3,4,1,2,3,4,5,6,7,8,9,10,5,6,7,8,9,10,11,12,13,14,11,12,13,14]
        ], dtype=object)
        
        # Spacing parameters - these control the tree layout
        self.delta_x = 4.0  # Horizontal distance between periods
        self.delta_y = 20.0 * spacing_multiplier  # Vertical spread
        
        self.x0, self.y0 = (0, 0)  # Starting position
        
        # Base box dimensions (will be scaled per period)
        self.base_box_x = 1.0
        self.base_box_y = 5
        
        # Base spacing between boxes
        self.base_box_spacing = spacing_multiplier
        
        # Plotting parameters
        self.plot_title = plot_title
        self.fontsize = 18
        self.labelsize = 20
        self.base_box_text_size = 15  # Base font size
        
        # Create period-specific box dimensions (after base_box_text_size is defined)
        self.box_dims_per_period = self._calculate_box_dimensions()
        self.figsize = figsize
        self.box_frame_color = 'black'
        self.facecolor = '#88E788'
        self.alpha = 0.6
        self.save_fig = save_fig
        self.fig_name = fig_name

        self.fig, self.ax = plt.subplots(1, figsize=self.figsize)

    def _calculate_box_dimensions(self):
        """Calculate box dimensions for each period based on scaling mode.
        
        Returns
        -------
        dict
            Dictionary with period as key and (box_x, box_y, font_size) as value
        """
        dims = {}
        
        for p in range(self.N_periods):
            # Calculate scaling factor based on mode
            if self.box_scale_mode == 'progressive':
                # Gradually decrease box size: period 0 largest, last period smallest
                scale_factor = 1.0 + (self.N_periods - 1 - p) * 0.3
            elif self.box_scale_mode == 'binary':
                # Two sizes: large for early periods, small for last
                scale_factor = 2.0 if p < self.N_periods - 1 else 1.0
            else:  # uniform
                scale_factor = 1.0
            
            box_x = self.base_box_x * scale_factor
            box_y = self.base_box_y * scale_factor
            
            # Calculate font size based on mode
            if self.font_scale_mode == 'progressive':
                # Smooth progression from large to small
                font_factor = 1.0 + (self.N_periods - 1 - p) * 0.4
            elif self.font_scale_mode == 'binary':
                # Two sizes: large for early, small for last
                font_factor = 3.0 if p < self.N_periods - 1 else 1.0
            else:  # uniform
                font_factor = 1.0
            
            font_size = self.base_box_text_size * font_factor
            
            dims[p] = (box_x, box_y, font_size)
        
        return dims

    def get_path_to_node(self, data):
        """Change path notation to node notation.

        This function takes data in path notation, i.e., with shape (32,6) and
        changes it to node notation, i.e., of shape (63,).

        Parameters
        ----------
        data: nd array
            data in path notation we're changing to node notation

        Returns
        -------
        node_data: nd array
            data in node notation
        """
        path_data = data
        N_paths = np.shape(path_data)[0]
        node_data = np.zeros(self.N_nodes)
        node_data_index = 0

        for i in range(0, self.N_periods):
            pts_taken_from_this_period = 2**i
            path_counter = int(N_paths/pts_taken_from_this_period)
            k = 0
            while k < N_paths:
                node_data[node_data_index] = path_data[k, i]
                k += path_counter
                node_data_index += 1

        return node_data

    def get_node_period(self, node_idx):
        """Determine which period a node belongs to.
        
        Parameters
        ----------
        node_idx: int
            Index of the node
            
        Returns
        -------
        int
            Period number (0-indexed)
        """
        cumulative_nodes = 0
        for p in range(self.N_periods):
            nodes_in_period = 2**p
            if node_idx < cumulative_nodes + nodes_in_period:
                return p
            cumulative_nodes += nodes_in_period
        return self.N_periods - 1

    def makePlot(self, pfig=False, show_fragility_arrow=True):
        """Make the tree plot.

        Parameters
        ----------
        pfig: bool
            if True, use special coloring to highlight path dependency
        show_fragility_arrow: bool
            if True, show the vertical "Increasing Fragility" arrow
        """
        # Set up colors
        if pfig:
            color_list = np.array([self.facecolor] * len(self.data))
            pink_inds = np.array([31, 32, 15])
            gold_inds = np.array([2, 5, 6, 10, 12, 13, 14, 19, 22, 24, 25, 27, 28, 29,
                         30, 36, 40, 43, 45, 46, 49, 51, 52, 54, 55,
                        56, 58, 59, 60, 61, 62])
            color_list[pink_inds] = "#CC79A7"
            color_list[gold_inds] = "#E69F00"
        else:
            color_list = [self.facecolor] * len(self.data)

        # Make coordinate values for nodes and edges
        self.make_node_coordinates()
        self.make_edge_coordinates()

        # Draw boxes and text at their designated locations
        for i in range(0, self.N_nodes):
            # Determine period for this node
            node_period = self.get_node_period(i)
            
            # Get dimensions for this period
            box_x, box_y, font_size = self.box_dims_per_period[node_period]
            
            # Create rectangle
            rect = Rectangle(self.node_coords[i], box_x, box_y,
                             edgecolor=self.box_frame_color,
                             facecolor=color_list[i], alpha=self.alpha,
                             zorder=2)
            self.ax.add_patch(rect)
            
            # Calculate center position for text
            text_x = self.node_coords[i, 0] + 0.5 * box_x
            text_y = self.node_coords[i, 1] + 0.5 * box_y
            
            # Format and draw text
            if self.is_cost:
                text_str = '${:.0f}'.format(self.data[i])
            else:
                text_str = '{:.0f}'.format(self.data[i])
            
            # Draw text with perfect centering
            # Use both ha='center' and va='center_baseline' for better vertical alignment
            self.ax.text(text_x, text_y, text_str, color='k',
                         fontsize=font_size, zorder=3,
                         ha='center', va='center_baseline', weight='bold')

        # Draw edges between boxes
        for i in range(0, self.N_nodes - 1):
            self.ax.plot(self.x_lines[i, :], self.y_lines[i, :], 'k', 
                        linewidth=0.8, zorder=1)

        # Set plot title only if provided
        if self.plot_title and self.plot_title.strip():
            self.fig.suptitle(self.plot_title, fontsize=self.fontsize, weight='bold')

        # Calculate axis ranges
        max_box_x = max([dims[0] for dims in self.box_dims_per_period.values()])
        max_box_y = max([dims[1] for dims in self.box_dims_per_period.values()])
        
        ymax = np.max(self.node_coords[:, 1]) + max_box_y + 2.0
        ymin = np.min(self.node_coords[:, 1]) - 3.0

        # Add fragility arrow if requested and calculate xmax accordingly
        if show_fragility_arrow:
            y_center = (ymin + ymax) / 2.0
            arrow_height = (ymax - ymin) * 0.80
            arrow_start = y_center - arrow_height / 2.0
            
            # Position arrow closer to the last period boxes
            last_period_box_x = self.box_dims_per_period[self.N_periods - 1][0]
            arrow_x = np.max(self.node_coords[:, 0]) + last_period_box_x + 0.8
            
            self.ax.arrow(arrow_x, arrow_start, 0, arrow_height, 
                         width=0.008, length_includes_head=True, 
                         head_width=0.3, head_length=2.5, color='k')
            self.ax.text(arrow_x + 0.5, y_center, "Increasing Fragility", 
                        rotation=270, fontsize=self.fontsize * 1.2, va='center')
            
            # Set xmax to end right after the arrow text with small margin
            xmax = arrow_x + 1.2
        else:
            # Without fragility arrow, end after the last period boxes
            xmax = np.max(self.node_coords[:, 0]) + max_box_x + 1.0

        self.ax.set_xlim((0, xmax))
        self.ax.set_ylim((ymin, ymax))

        # Set x-axis label
        self.ax.set_xlabel("Year", fontsize=self.fontsize * 1.5, weight='bold')

        # Turn off y-axis and borders
        self.ax.get_yaxis().set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # Set up x-axis ticks and labels
        self.ax.tick_params(axis='x', labelsize=self.labelsize)
        
        # Generate tick positions centered on boxes for each period
        x_tick_positions = []
        for i in range(self.N_periods):
            box_x, _, _ = self.box_dims_per_period[i]
            x_tick_positions.append(i * self.delta_x + 0.5 * box_x)
        
        self.ax.set_xticks(x_tick_positions)
        
        # Generate year labels from decision_times
        year_labels = []
        for i in range(self.N_periods):
            if i < len(self.decision_times):
                year_labels.append(str(self.base_year + self.decision_times[i]))
            else:
                year_labels.append('')
        self.ax.set_xticklabels(year_labels, fontsize=self.fontsize * 1.5)

        # Add top x-axis with period numbers
        top_x = self.ax.twiny()
        top_x.set_xlim((0, xmax))
        top_x.set_xticks(x_tick_positions)
        
        period_labels = [str(i) for i in range(self.N_periods)]
        top_x.set_xticklabels(period_labels, fontsize=self.fontsize * 1.5)
        top_x.spines['left'].set_visible(False)
        top_x.spines['right'].set_visible(False)
        top_x.spines['bottom'].set_visible(False)
        top_x.set_xlabel("Period", labelpad=8.0, fontsize=self.fontsize * 1.5, weight='bold')

        # Finalize layout
        self.fig.tight_layout()

        if self.save_fig:
            self.fig.savefig(self.fig_name + ".png", dpi=400, bbox_inches='tight')

    def make_node_coordinates(self):
        """Create coordinate positions for all nodes in the tree."""
        N_points_covered = 0

        for period in range(0, self.N_periods):
            # Get box dimensions for this period
            box_x, box_y, _ = self.box_dims_per_period[period]
            
            # Number of points in this period
            N_pts_period = 2**period

            # X coordinates (horizontal position)
            self.node_coords[N_points_covered:N_points_covered+N_pts_period, 0] = \
                period * self.delta_x

            # Check if this is the last period and we want even spacing
            if period == self.N_periods - 1 and self.even_spacing_last_period:
                # Even spacing for last period
                # Calculate total vertical span
                y_max = period * self.delta_y
                y_min = -period * self.delta_y
                total_span = y_max - y_min
                
                # Evenly distribute boxes across the span
                if N_pts_period > 1:
                    spacing = total_span / (N_pts_period - 1)
                    for i in range(N_pts_period):
                        # Position from top to bottom, adjusted for box anchor point
                        self.node_coords[N_points_covered + i, 1] = \
                            y_max - i * spacing - 0.5 * box_y
                else:
                    # Single box, center it
                    self.node_coords[N_points_covered, 1] = -0.5 * box_y
            else:
                # Normal grouped structure
                # Number of groups
                N_groups = period + 1

                # Interior points and groups
                N_interior_pts = N_pts_period - 2
                N_interior_groups = N_groups - 2

                # Y coordinates (vertical position)
                # Top-most node
                self.node_coords[N_points_covered, 1] = \
                    period * self.delta_y - 0.5 * box_y

                # Bottom-most node
                self.node_coords[N_points_covered + N_pts_period - 1, 1] = \
                    -period * self.delta_y - 0.5 * box_y

                # Interior nodes
                self.make_group_coords(period, N_points_covered, N_interior_groups, 
                                      N_interior_pts, box_y)

            N_points_covered += N_pts_period

    def make_group_coords(self, period, N_points_covered, N_interior_groups,
                          N_interior_points, box_y):
        """Make coordinate values for grouped boxes.
        
        Parameters
        ----------
        period: int
            the current period
        N_points_covered: int
            number of points already assigned locations
        N_interior_groups: int
            number of interior groups in the period
        N_interior_points: int
            number of interior points in the period
        box_y: float
            height of boxes in this period
        """
        if N_interior_groups > 0:
            center_points = np.zeros(N_interior_groups)

            # First group centered on uppermost box from two periods ago
            center_points[0] = self.delta_y * (period - 2)

            # Last group centered on lowest box from two periods ago
            center_points[-1] = -1 * self.delta_y * (period - 2)

            # Partitioning of interior nodes
            partitioning = self.get_partitioning(period, N_interior_points,
                                                N_interior_groups)

            # Special cases for periods 4 and 5
            if N_interior_groups == 3:
                center_points[1] = self.delta_y * (period - 4)
            elif N_interior_groups == 4:
                center_points[1] = self.delta_y * (period - 4)
                center_points[2] = -1 * self.delta_y * (period - 4)

            tmp_nodes_covered = N_points_covered + 1

            # Assign coordinates to each interior group
            for i in range(N_interior_groups):
                N_pts_group = int(partitioning[i])

                if N_pts_group % 2 == 0:
                    # Even number: split around center
                    ref_y = center_points[i] - box_y - 0.5 * self.base_box_spacing
                    N_pts_above = N_pts_group // 2 + 1
                    N_pts_below = N_pts_group // 2 - 1

                    for j in range(0, N_pts_above):
                        tmp_dist = ref_y + (N_pts_above - 1 - j) * \
                                  (box_y + self.base_box_spacing)
                        self.node_coords[tmp_nodes_covered, 1] = tmp_dist
                        tmp_nodes_covered += 1

                    for j in range(0, N_pts_below):
                        tmp_dist = ref_y - (j + 1) * \
                                  (box_y + self.base_box_spacing)
                        self.node_coords[tmp_nodes_covered, 1] = tmp_dist
                        tmp_nodes_covered += 1

                else:
                    # Odd number: center box on center point
                    ref_y = center_points[i] - 0.5 * box_y
                    N_pts_above = N_pts_group // 2 + 1
                    N_pts_below = N_pts_group // 2

                    for j in range(0, N_pts_above):
                        tmp_dist = ref_y + (N_pts_above - 1 - j) * \
                                  (box_y + self.base_box_spacing)
                        self.node_coords[tmp_nodes_covered, 1] = tmp_dist
                        tmp_nodes_covered += 1

                    for j in range(0, N_pts_below):
                        tmp_dist = ref_y - (j + 1) * \
                                  (box_y + self.base_box_spacing)
                        self.node_coords[tmp_nodes_covered, 1] = tmp_dist
                        tmp_nodes_covered += 1

    def make_edge_coordinates(self):
        """Create coordinates for edges connecting nodes."""
        indexes_covered = 1

        # Set end points for all edges (connecting to current period nodes)
        for i in range(1, self.N_nodes):
            # Get period and box dimensions for this node
            node_period = self.get_node_period(i)
            _, box_y, _ = self.box_dims_per_period[node_period]
            
            # End point is left edge of box, centered vertically
            self.x_lines[i-1, 1] = self.node_coords[i, 0]
            self.y_lines[i-1, 1] = self.node_coords[i, 1] + 0.5 * box_y

        # Set start points for all edges (connecting from previous period nodes)
        for p in range(1, self.N_periods):
            N_pts_current = 2**p
            N_pts_former = 2**(p-1)
            
            # Get box dimensions for previous and current periods
            prev_box_x, prev_box_y, _ = self.box_dims_per_period[p-1]
            _, curr_box_y, _ = self.box_dims_per_period[p]

            # X-line start points (right edge of previous period's boxes)
            for node in range(N_pts_current):
                tmp_index = indexes_covered + node - 1
                
                # Determine parent node index
                if node == 0 or node == 1:
                    parent_idx = indexes_covered - N_pts_former
                elif node == N_pts_current - 1 or node == N_pts_current - 2:
                    parent_idx = indexes_covered - 1
                else:
                    if p-3 < len(self.track_additions):
                        additions = self.track_additions[p-3]
                        parent_idx = indexes_covered - N_pts_former + additions[node-2]
                    else:
                        parent_offset = (node - 2) // 2
                        if parent_offset >= N_pts_former:
                            parent_offset = N_pts_former - 1
                        parent_idx = indexes_covered - N_pts_former + parent_offset
                
                # Start point is right edge of parent box, centered vertically
                self.x_lines[tmp_index, 0] = self.node_coords[parent_idx, 0] + prev_box_x
                self.y_lines[tmp_index, 0] = self.node_coords[parent_idx, 1] + 0.5 * prev_box_y

            indexes_covered += N_pts_current

    def get_partitioning(self, period, N_interior_pts, N_interior_groups):
        """Create partitioning of interior group points.

        Parameters
        ----------
        period: int
            current period
        N_interior_pts: int
            number of interior points
        N_interior_groups: int
            number of interior groups

        Returns
        -------
        ndarray
            partitioning array
        """
        if N_interior_groups > 0:
            partitioning = np.zeros(N_interior_groups)

            partitioning[0] = period
            partitioning[-1] = period

            N_remaining_interior_pts = N_interior_pts - 2 * period
            N_remaining_groups = N_interior_groups - 2

            if N_remaining_groups == 1:
                partitioning[1] = N_remaining_interior_pts
            elif N_remaining_groups == 2:
                partitioning[1:3] = int(N_remaining_interior_pts/2)

            return partitioning
