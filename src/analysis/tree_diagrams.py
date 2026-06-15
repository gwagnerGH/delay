
"""Tree diagram class.

Adam Michael Bauer
University of Illinois at Urbana Champaign
adammb4@illinois.edu
5/18/2022

This code contains the TreeDiagram class, which conviniently creates tree
diagrams of the output of CAP6.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.patches import Rectangle

from ..config import DEFAULT_BASE_YEAR, DEFAULT_DECISION_TIMES

mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

class TreeDiagram:
    """Tree diagram class.

    Makes plot of CAP6 model output in path dependent binomial tree
    form.

    Parameters
    ----------
    N_periods: int
        number of periods
    data: nd array
        the data we want plotted
    path_notation: bool
        is the data in path notation? i.e., does the data have shape (32,6)?
    plot_title: string
        title of plot
    figsize: tuple
        size of figure
    save_fig: bool
        dictates whether we should save the figure or not
    fig_name: string
        filename if save fig is turned on
    is_cost: bool
        if True, puts a little dollar sign in front of the text in the boxes.

    Attributes
    ----------
    N_periods: int
        number of periods
    N_nodes: int
        number of nodes in the tree; if p is number of periods, number of nodes
        is 2^(p - 1).
    path_notation: bool
        is the data in path notation? i.e., True if data has shape (32,6)
    data: nd array
        data being plotted
    node_coords: nd array
        list of tuples telling the location of nodes in the tree
    x_lines: nd array
        distance between boxes in the x direction. the 1st (2nd, resp.) element
        of the ordered pair is the start (finish, resp.) of the line
    y_lines: nd array
        same as x_lines, but in y dimension
    track_additions: nd array
        slop. this is for referene in drawing tree edges. this is the only
        attribute of the class that is not general; it is hardwired into the
        code. it tells which nodes to draw edges to and from for later sections
        of the tree.
    delta_x: flaot
        step for node locations in x
    delta_y: float
        step for node locations in y
    x0: float
        start location for first node in x
    y0: float
        start location for first node in y
    box_x: float
        box width
    box_y: float
        box height
    box_spacing: float
        spacing bewteen boxes (vertically)
    plot_title: string
        name of plot
    fontsize: float
        fontsize of axis labels and legend
    labelszie: float
        size of tick marks
    box_text_size: float
        size of text within boxes at each node
    figsize: tuple
        matplotlib figsize param, tells how big the plot is
    box_frame_color: string
        color of exterior of rectangle
    facecolor: string
        color of interior of rectange
    alpha: float
        transparency value of box
    save_fig: bool
        T/F: are we saving the figure?
    fig_name: string
        document name if we're saving the figure
    fig, ax: `matplotlib.pyplot.subplots` objects
        conventional fig, ax for plotting in matplotlib
    is_cost: bool
        if True, puts a little dollar sign in front of the text in the boxes.
    """

    def __init__(self, N_periods, data, path_notation, plot_title,
                 figsize=(12,8), save_fig=True, fig_name='tree_plot',
                 is_cost=False, decision_times=None, base_year=DEFAULT_BASE_YEAR):
        self.N_periods = N_periods
        self.N_nodes = int(2**self.N_periods - 1)
        self.path_notation = path_notation
        self.is_cost = is_cost
        
        # Store decision times for custom year labels
        if decision_times is None:
            self.decision_times = DEFAULT_DECISION_TIMES
        else:
            self.decision_times = decision_times
        self.base_year = base_year

        """if the data is in path notation, i.e., is (32,6), then run
        getPathToNode to change the shape to (63,) (i.e., node notation).
        """
        if self.path_notation:
            self.data = self.get_path_to_node(data)
        else:
            self.data = data

        """To do with: box and edge oordinate values.
        """

        self.node_coords = np.zeros((self.N_nodes, 2))
        self.x_lines = np.zeros((self.N_nodes-1, 2))
        self.y_lines = np.zeros((self.N_nodes-1, 2))
        self.track_additions = np.array([[1,2,1,2], [1,2,3,1,2,3,4,5,6,4,5,6],
                                         [1,2,3,4,1,2,3,4,5,6,7,8,9,10,5,6,7,8,9,10,11,12,13,14,11,12,13,14]],
                                        dtype=object)
        
        # Spacing parameters - adjust these to control tree layout
        self.delta_x = 3          # Horizontal distance between periods (tree levels)
        self.delta_y = 18        # Vertical distance between extreme nodes in each period
                                  # Increase this if boxes overlap vertically (default was 12)
        
        self.x0, self.y0 = (0,0)  # Starting position for the first node
        
        # Box dimensions - adjust these to change node appearance
        self.box_x = 1.5            # Box width (decrease to make boxes narrower)
        self.box_y = 2.0           # Box height (increase to make boxes taller, default was 2)
        
        self.box_spacing = 1.0   # Minimum vertical gap between adjacent boxes within groups
                                  # Increase this if boxes overlap (default was 0.35)

        """To do with: plotting.
        """
        self.plot_title = plot_title
        self.fontsize = 22
        self.labelsize = 20
        self.box_text_size = 9
        self.figsize = figsize
        self.box_frame_color = 'black'
        self.facecolor = '#56B4E9'
        self.alpha = 0.6
        self.save_fig = save_fig
        self.fig_name = fig_name

        self.fig, self.ax = plt.subplots(1, figsize=self.figsize)

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

        # take data in path notation and note some of its important attributes
        path_data = data
        N_paths = np.shape(path_data)[0] # number of paths in data
        node_data = np.zeros(self.N_nodes)
        node_data_index = 0

        # loop through the number of periods, and extract only the
        # non-redundant values in the path notation data.
        # these are the values for each individual node.
        for i in range(0, self.N_periods):
            pts_taken_from_this_period = 2**i # take this many points from each period
            # this many relevant points are in each period
            path_counter = int(N_paths/pts_taken_from_this_period)
            k = 0
            while k < N_paths:
                # for example, node_data[0] = path_data[0,0]
                # node_data[1] = path_data[0,1].
                # node_data[2] = path_data[16,1], and so on.
                node_data[node_data_index] = path_data[k,i]
                k += path_counter # index path_data values
                node_data_index += 1 # index node_data 

        return node_data

    def makePlot(self, pfig=False):
        """Make plotting attributes.

        This function makes the tree plot. This is where the magic happens!

        Parameters
        ----------
        pfig: bool
            in our paper, we color certain boxes differently to highlight path
            dependency in the tree structure. turn this on if you'd like that
            different coloring to occur.
        """

        if pfig:
            color_list = np.array([self.facecolor] * len(self.data))
            pink_inds = np.array([31, 32, 15])
            gold_inds = np.array([2, 5, 6, 10, 12, 13, 14, 19, 22, 24, 25, 27, 28, 29,\
                         30, 36, 40, 43, 45, 46, 49, 51, 52, 54, 55,\
                        56, 58, 59, 60, 61, 62])
            color_list[pink_inds] = "#CC79A7"
            color_list[gold_inds] = "#E69F00"
        else:
            color_list = [self.facecolor] * len(self.data)

        # make coordinate values for nodes and edges
        self.make_node_coordinates()
        self.make_edge_coordinates()

        # draw boxes at their designated locations
        for i in range(0, self.N_nodes):
            # Determine which period this node belongs to
            # Period p has nodes from 2^p - 1 to 2^(p+1) - 2
            node_period = 0
            cumulative_nodes = 0
            for p in range(self.N_periods):
                nodes_in_period = 2**p
                if i < cumulative_nodes + nodes_in_period:
                    node_period = p
                    break
                cumulative_nodes += nodes_in_period
            
            # For prices (is_cost=True), use larger boxes for periods 0-4
            if self.is_cost and node_period < self.N_periods - 1:
                # Periods 0-4: use larger boxes (2.5x width, 2x height)
                current_box_x = self.box_x * 1.5
                current_box_y = self.box_y * 2.0
            else:
                # Period 5 or non-cost diagrams: use normal box size
                current_box_x = self.box_x
                current_box_y = self.box_y
            
            rect = Rectangle(self.node_coords[i], current_box_x, current_box_y,
                             edgecolor=self.box_frame_color,
                             facecolor=color_list[i], alpha=self.alpha,
                             zorder=2)
            self.ax.add_patch(rect)
            
            # Calculate center position of the box for text placement
            text_x = self.node_coords[i,0] + 0.5 * current_box_x  # Horizontal center
            text_y = self.node_coords[i,1] + 0.5 * current_box_y  # Vertical center
            
            # For prices (is_cost=True), use 3x font size for periods 0-4, normal for period 5
            if self.is_cost:
                if node_period < self.N_periods - 1:  # Periods 0-4
                    text_fontsize = (self.box_text_size - 0.5) * 3
                else:  # Period 5 (last period)
                    text_fontsize = self.box_text_size - 0.5
                
                self.ax.text(text_x, text_y,
                             '${:0.2f}'.format(self.data[i]), color='k',
                             fontsize=text_fontsize, zorder=3,
                             ha='center', va='center')  # Center alignment
            else:
                self.ax.text(text_x, text_y,
                             '{:0.2f}'.format(self.data[i]), color='k',
                             fontsize=self.box_text_size-0.5, zorder=3,
                             ha='center', va='center')  # Center alignment

        # draw edges between boxes
        for i in range(0, self.N_nodes - 1):
            self.ax.plot(self.x_lines[i,:], self.y_lines[i,:], 'k', zorder=1)
        
        # Add continuation arrows if there are more periods than displayed
        # Check if we're truncating the tree (i.e., there are more decision times available)
        total_periods_available = len(self.decision_times)
        if self.N_periods + 1 < total_periods_available:
            # Add small horizontal arrows to the right of the last period's boxes
            # to indicate continuation
            last_period_start_node = 2**(self.N_periods - 1) - 1
            last_period_end_node = 2**self.N_periods - 1
            
            for node_idx in range(last_period_start_node, last_period_end_node):
                # Get the position of each box in the last period
                box_x_end = self.node_coords[node_idx, 0] + self.box_x
                box_y_center = self.node_coords[node_idx, 1] + 0.5 * self.box_y
                
                # Draw a small horizontal arrow
                arrow_length = 0.8
                self.ax.arrow(box_x_end + 0.1, box_y_center, arrow_length, 0,
                            width=0.08, length_includes_head=True,
                            head_width=0.6, head_length=0.2,
                            color='black', alpha=0.7, zorder=2)

        # set plot title
        self.fig.suptitle(self.plot_title, fontsize=self.fontsize)

        # set x and y axis ranges
        # add a little to end of xmax to fit the arrow
        xmax = np.max(self.node_coords[:, 0]) + self.box_x + 2.2
        ymax = np.max(self.node_coords[:, 1]) + self.box_y
        ymin = np.min(self.node_coords[:, 1]) - 2

        self.ax.set_xlim((0, xmax))
        self.ax.set_ylim((ymin, ymax+1.5))

        # add arrow upwards showing increasing direction of fragility; add x
        # axis label
        # Calculate center and height for proper vertical centering
        y_center = (ymin + ymax) / 2.0
        arrow_height = (ymax - ymin) * 0.85  # Use 85% of available height
        arrow_start = y_center - arrow_height / 2.0
        
        self.ax.arrow(xmax - 1, arrow_start, 0, arrow_height, width=0.005,
                      length_includes_head=True, head_width=0.25, head_length=2,
                      color='k')
        self.ax.text(xmax-.65, y_center, "Increasing Fragility", rotation=270,
                     fontsize=self.fontsize, va='center')
        self.ax.set_xlabel("Year", fontsize=self.fontsize)

        # turn off y axis and left, top, and right borders off
        self.ax.get_yaxis().set_visible(False)
        self.ax.spines['left'].set_visible(False)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # change x tick sizes as well as their labels
        # Dynamically generate x-axis ticks based on N_periods
        self.ax.tick_params(axis='x', labelsize=self.labelsize)
        
        # Generate tick positions for each period
        x_tick_positions = [i * self.delta_x + 0.5 * self.box_x for i in range(self.N_periods)]
        self.ax.set_xticks(x_tick_positions)
        
        # Generate year labels from decision_times
        # Use the actual decision times provided to generate year labels
        year_labels = []
        for i in range(self.N_periods):
            if i < len(self.decision_times):
                year_labels.append(str(self.base_year + self.decision_times[i]))
            else:
                # Fallback if not enough decision times provided
                year_labels.append('')
        self.ax.set_xticklabels(year_labels)

        # put x axis on top with different labels (period numbers)
        top_x = self.ax.twiny()
        top_x.set_xlim((0, xmax))
        top_x.set_xticks(x_tick_positions)
        
        # Generate period labels (0, 1, 2, ..., N_periods-1)
        period_labels = [str(i) for i in range(self.N_periods)]
        top_x.set_xticklabels(period_labels)
        top_x.spines['left'].set_visible(False)
        top_x.spines['right'].set_visible(False)
        top_x.spines['bottom'].set_visible(False)
        top_x.set_xlabel("Period", labelpad=8.0)

        # tighten things up
        self.fig.tight_layout()

        if self.save_fig:
            self.fig.savefig(self.fig_name+".png", dpi=400)

    def make_node_coordinates(self):
        """Make node coordaintes.

        This function creates the attribute node_coords, which locates the
        boxes that display text in the tree plot.
        """

        # tracks how many points we have values for already
        N_points_covered = 0

        for period in range(0, self.N_periods):
            # number of points in this period
            N_pts_period = 2**period

            # number of "groups", meaning the number of individual values for
            # fragility in a given period
            N_groups = period + 1

            # number of points not considering the outside-most two
            N_interior_pts = N_pts_period - 2

            # number of groups not considering the outside-most two
            N_interior_groups = N_groups - 2

            # x points (easy)
            self.node_coords[N_points_covered:N_points_covered+N_pts_period,0]\
                = period * self.delta_x

            # y points (hard)
            # impo: I want the center of each box to happen at period * self.delta_y. 
            # however, i plan on using the rectangle patch in matplotlib to plot each box.
            # therefore, the locations i provide in self.node_coords must locate not the 
            # center of the box, but the lower left corner of it, by the definition
            # of the anchor point per the matplotlib documentation. this justifies the 
            # - 0.5 * self.box_y at the end of these.

            # largest y val is the first node in the period 
            self.node_coords[N_points_covered, 1] = period * self.delta_y\
                                                    - 0.5 * self.box_y

            # smallest (most negative) y val is the last node in the period
            self.node_coords[N_points_covered+N_pts_period-1, 1] = - period\
                                                                   * self.delta_y\
                                                                   - 0.5 * self.box_y

            # for all the nodes in between :)
            self.make_group_coords(period, N_points_covered, N_interior_groups, N_interior_pts)

            # increment points covered after every node in this period has been
            # assigned a location
            N_points_covered += N_pts_period

    def make_group_coords(self, period, N_points_covered, N_interior_groups,
                        N_interior_points):
        """Make coordinate values for group boxes.

        A group is a set of boxes which are grouped together in the tree. This
        function creates the coordinates of each box in these groups for some
        period.

        Parameters
        ----------
        period: int
            the period we're in
        N_points_covered: int
            the number of points we've already assigned locations for in the
            period at hand
        N_interior_groups: int
            the number of groups in the period
        N_interior_points: int
            the number of points in the interior of the current period; this is
            equal to the total number of points in the period minus 2, the
            outside most points that aren't a member of a group.
        """

        if N_interior_groups > 0:
            # create center points for each group's coordinates
            center_points = np.zeros(N_interior_groups)

            # first group (from the top) is centered on the upper most box two
            # period ago
            center_points[0] = self.delta_y * (period - 2)

            # last group (from the top) is centered on the lowest box two
            # periods ago
            center_points[-1] = - 1 * self.delta_y * (period - 2)

            # create partitioning of interior node locations
            partitioning = self.get_partitioning(period, N_interior_points,
                                                N_interior_groups)

            # if we're in the fourth period, there is one extra group (of 6
            # elements). center this on the high point four periods ago.
            if N_interior_groups == 3:
                center_points[1] = self.delta_y * (period - 4)

            # if we're in the fifth and final period, we have two extra groups,
            # which are centered on the highest and lowest four periods ago
            elif N_interior_groups == 4:
                center_points[1] = self.delta_y * (period - 4)
                center_points[2] = -1 * self.delta_y * (period - 4)

            # temporary tracker of how many nodes we've gone through
            tmp_nodes_covered = N_points_covered + 1

            # now loop through each interior group and assign it coordinates
            for i in range(N_interior_groups):
                # number of points in group i
                N_pts_group = int(partitioning[i])

                # if the number of points in the group is even, the boxes will
                # be split around the center (i.e., two above and two below)
                if N_pts_group % 2 == 0:
                    # reference center line, offset by box width + half a
                    # spacing between the boxes so that our reference position
                    # is the lower left corner

                    # of the box closest to center (draw it out!)
                    ref_y = center_points[i] - self.box_y\
                            - 0.5 * self.box_spacing

                    # number of points above and below center line
                    N_pts_above = N_pts_group // 2 + 1
                    N_pts_below = N_pts_group // 2 - 1

                    # now assign positions to all points above center line
                    for j in range(0, N_pts_above):
                        tmp_dist = ref_y + (N_pts_above - 1 - j)\
                                 * (self.box_y + self.box_spacing)
                        self.node_coords[tmp_nodes_covered, 1] = tmp_dist
                        tmp_nodes_covered += 1

                    # now assign positions to all points below center line
                    for j in range(0, N_pts_below):
                        tmp_dist = ref_y - (j + 1)\
                                 * (self.box_y + self.box_spacing)
                        self.node_coords[tmp_nodes_covered, 1] = tmp_dist
                        tmp_nodes_covered += 1

                # if odd, one box will have its center on the center point of
                # the group.
                else:
                    ref_y = center_points[i] - 0.5 * self.box_y
                    N_pts_above = N_pts_group // 2 + 1
                    N_pts_below = N_pts_group // 2

                    # now assign positions to all points above center line
                    for j in range(0, N_pts_above):
                        tmp_dist = ref_y + (N_pts_above - 1 - j)\
                                 * (self.box_y + self.box_spacing)
                        self.node_coords[tmp_nodes_covered, 1] = tmp_dist
                        tmp_nodes_covered += 1

                    # now assign positions to all points below center line
                    for j in range(0, N_pts_below):
                        tmp_dist = ref_y - (j + 1)\
                                 * (self.box_y + self.box_spacing)
                        self.node_coords[tmp_nodes_covered, 1] = tmp_dist
                        tmp_nodes_covered += 1

    def make_edge_coordinates(self):
        """Make edge coordinates.

        Make list of coordinates to draw the edges, or the lines, that connect
        each box in the tree.
        """

        # number of indexes which have values
        indexes_covered = 1

        # end points in x are just the location of each box; end points in y
        # are the end point plus a shift
        self.x_lines[:, 1] = self.node_coords[1:, 0]
        self.y_lines[:, 1] = self.node_coords[1:, 1] + 0.5 * self.box_y

        # loop through periods and make the edge coordinates
        for p in range(1, self.N_periods):
            # points in current period and previous period
            N_pts_current = 2**p
            N_pts_former = 2**(p-1)

            # x line start points are just the end points of the previous
            # period's nodes
            self.x_lines[indexes_covered-1:indexes_covered+N_pts_current-1, 0]\
                = self.node_coords[indexes_covered-1, 0] + self.box_x

            # loop through nodes in current period
            for node in range(N_pts_current):
                # index of current node (in (63,) notation)
                tmp_index = indexes_covered + node - 1

                # if first or second node (from the top), then easily connect
                # to first point in previous period
                if node == 0 or node == 1:
                    self.y_lines[tmp_index, 0] = self.node_coords[indexes_covered-N_pts_former, 1]\
                                               + 0.5 * self.box_y

                # if last or second last node (from the top), then easily
                # connect to last point in previous period
                elif node == N_pts_current - 1 or node == N_pts_current - 2:
                    self.y_lines[tmp_index, 0] = self.node_coords[indexes_covered-1, 1]\
                                               + 0.5 * self.box_y

                # interior points are dictated by the additions i laid out a
                # priori for periods 3-5 (p-3 = 0-2)
                # For periods beyond this, we need to compute connections dynamically
                else:
                    if p-3 < len(self.track_additions):
                        # Use pre-defined additions for periods 3-5
                        additions = self.track_additions[p-3]
                        self.y_lines[tmp_index, 0] = self.node_coords[indexes_covered-N_pts_former+additions[node - 2], 1]\
                                                   + 0.5 * self.box_y
                    else:
                        # For periods 6+, compute connection dynamically
                        # Connect interior nodes to their corresponding parent node
                        # based on the binary tree structure
                        parent_offset = (node - 2) // 2
                        if parent_offset >= N_pts_former:
                            parent_offset = N_pts_former - 1
                        self.y_lines[tmp_index, 0] = self.node_coords[indexes_covered-N_pts_former+parent_offset, 1]\
                                                   + 0.5 * self.box_y

            # add the points in this period to the indexes we've covered
            indexes_covered += N_pts_current

    def get_partitioning(self, period, N_interior_pts, N_interior_groups):
        """Make partitioning of interior group points.

        This function creates a "partitioning," dictating how many individual
        boxes are in each group in the period. The values are ordered from the
        top to the bottom of the tree in each period.

        Parameters
        ----------
        period: int
            the current period
        N_interior_points: int
            the number of interior points
        N_interior_groups: int
            the number of interior groups in the period

        Returns
        -------
        partitioning: list
            list dictating the number of nodes in each interior group for a
            given period.
        """

        if N_interior_groups > 0:
            partitioning = np.zeros(N_interior_groups)

            # top and bottom groups have the same number of points as the
            # current period
            partitioning[0] = period
            partitioning[-1] = period

            # remaining interior points and groups after the top and
            # bottommoost points are accounted for
            N_remaining_interior_pts = N_interior_pts - 2 * period
            N_remaining_groups = N_interior_groups - 2

            # if there is one group remaining, assign all the remaining points
            # to that group
            if N_remaining_groups == 1:
                partitioning[1] = N_remaining_interior_pts

            # if there are two remaining groups, assign half to one and half to
            # the other
            elif N_remaining_groups == 2:
                partitioning[1:3] = int(N_remaining_interior_pts/2)

            return partitioning
