import base64
import io
from typing import Dict, List, Tuple

import matplotlib

# Ensure a non-interactive backend for headless environments.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

class DynamicGraph:
    """Minimal NetworkX directed graph builder that returns mind map images."""
    
    def __init__(self, node_names: List[str], edge_map: List[Tuple[str, str]]):
        self.node_names = node_names
        self.edge_map = edge_map
        self.graph = None
        
    def build_graph(self) -> nx.DiGraph:
        """Create and return the directed NetworkX graph."""
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.node_names)
        self.graph.add_edges_from(self.edge_map)
        return self.graph
    
    def generate_image(self, save_path: str = None, **kwargs) -> Image.Image:
        """Generate and return the mind map as a PIL Image."""
        if not self.graph:
            self.build_graph()
        
        return self._create_graph_image(self.node_names, self.edge_map, save_path, **kwargs)
    
    def get_base64_image(self, **kwargs) -> str:
        """Generate mind map and return as base64 encoded string."""
        image = self.generate_image(**kwargs)
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    @staticmethod
    def create_graph(nodes: List[str], edges: List[Tuple[str, str]], **kwargs) -> str:
        """
        Create a mind map graph and return base64 encoded data URL.
        
        Args:
            nodes: List of node names
            edges: List of edge tuples (from_node, to_node)
            **kwargs: Styling parameters
            
        Returns:
            Base64 encoded data URL string (data:image/png;base64,...)
        """
        image = DynamicGraph._create_graph_image(nodes, edges, **kwargs)
        
        # Convert to base64 data URL
        buf = io.BytesIO()
        image.save(buf, format='PNG')
        buf.seek(0)
        base64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{base64_string}"
    
    @staticmethod
    def _create_graph_image(nodes: List[str], edges: List[Tuple[str, str]], 
                           save_path: str = None, **kwargs) -> Image.Image:
        """Internal method to create graph image."""
        
        # Create graph
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)
        
        # Create figure
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 8)))
        pos = nx.spring_layout(graph, k=2, iterations=50)
        
        # Draw edges first
        nx.draw_networkx_edges(
            graph, pos, 
            arrows=True,
            arrowsize=kwargs.get('arrowsize', 30),
            arrowstyle=kwargs.get('arrowstyle', '->'),
            edge_color=kwargs.get('edge_color', 'darkblue'),
            width=kwargs.get('edge_width', 2),
            alpha=kwargs.get('edge_alpha', 0.7),
            connectionstyle="arc3,rad=0.1",
            ax=ax
        )
        
        # Draw nodes on top
        nx.draw_networkx_nodes(
            graph, pos, 
            node_color=kwargs.get('node_color', 'lightgreen'),
            node_size=kwargs.get('node_size', 2000),
            alpha=kwargs.get('node_alpha', 0.9),
            edgecolors='black',
            linewidths=2,
            ax=ax
        )
        
        # Draw labels
        nx.draw_networkx_labels(
            graph, pos, 
            font_size=kwargs.get('font_size', 11),
            font_weight='bold',
            font_color='black',
            ax=ax
        )
        
        ax.set_title(kwargs.get('title', 'Mind Map'), fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=kwargs.get('dpi', 300), 
                   bbox_inches='tight', facecolor='white')
        buf.seek(0)
        image = Image.open(buf)
        
        # Save if path provided
        if save_path:
            image.save(save_path)
            
        plt.close(fig)  # Clean up
        return image
    
    def get_info(self) -> Dict:
        """Return basic directed graph information."""
        if not self.graph:
            self.build_graph()
        
        return {
            'nodes': len(self.graph.nodes()),
            'edges': len(self.graph.edges()),
            'density': nx.density(self.graph),
            'weakly_connected': nx.is_weakly_connected(self.graph),
            'strongly_connected': nx.is_strongly_connected(self.graph)
        }

# Usage examples
def demo_create_graph_function():
    """Demonstrate the static create_graph function."""
    
    # Example 1: C Pointers Mind Map
    c_pointers_nodes = [
        'C Pointers', 'Declaration', 'Operations', 'Advanced Topics',
        'int *ptr', 'char *str', 'Dereferencing', 'Address-of', 
        'Arithmetic', 'Double Pointers', 'Function Pointers', 'Memory Management'
    ]
    
    c_pointers_edges = [
        ('C Pointers', 'Declaration'),
        ('C Pointers', 'Operations'),
        ('C Pointers', 'Advanced Topics'),
        ('Declaration', 'int *ptr'),
        ('Declaration', 'char *str'),
        ('Operations', 'Dereferencing'),
        ('Operations', 'Address-of'),
        ('Operations', 'Arithmetic'),
        ('Advanced Topics', 'Double Pointers'),
        ('Advanced Topics', 'Function Pointers'),
        ('Advanced Topics', 'Memory Management')
    ]
    
    # Generate base64 data URL - ready for web usage
    data_url = DynamicGraph.create_graph(
        nodes=c_pointers_nodes,
        edges=c_pointers_edges,
        node_color='lightgreen',
        node_size=2500,
        figsize=(14, 10),
        arrowsize=25,
        edge_color='darkblue',
        edge_width=2.5,
        font_size=10,
        title='C Pointers Mind Map'
    )
    
    # Example 2: Simple concept map
    simple_nodes = ['Programming', 'Variables', 'Functions', 'Loops']
    simple_edges = [('Programming', 'Variables'), ('Programming', 'Functions'), ('Programming', 'Loops')]
    
    simple_data_url = DynamicGraph.create_graph(
        nodes=simple_nodes,
        edges=simple_edges,
        node_color='lightcoral',
        title='Programming Basics'
    )
    
    return data_url, simple_data_url

# Quick helper function for even simpler usage
def quick_mindmap(nodes: List[str], edges: List[Tuple[str, str]], title: str = "Mind Map") -> str:
    """Quick function to create a mind map with default styling."""
    return DynamicGraph.create_graph(
        nodes=nodes,
        edges=edges,
        node_color='lightblue',
        node_size=2000,
        title=title,
        figsize=(12, 8)
    )

# Usage
if __name__ == "__main__":
    print("Demonstrating create_graph static function...")
    
    # Get data URLs
    c_url, simple_url = demo_create_graph_function()
    
    print(f"C Pointers data URL length: {len(c_url)} characters")
    print(f"Simple map data URL length: {len(simple_url)} characters")
    print(f"Data URL starts with: {c_url[:50]}...")
    
    # Quick usage example
    nodes = ['Study', 'Read', 'Practice', 'Review']
    edges = [('Study', 'Read'), ('Study', 'Practice'), ('Study', 'Review')]
    study_url = quick_mindmap(nodes, edges, "Study Process")
    
    print(f"Quick mindmap URL length: {len(study_url)} characters")
    
    # These data URLs can be used directly in HTML img tags:
    # <img src="{data_url}" alt="Mind Map" />
